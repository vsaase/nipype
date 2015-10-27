import inspect
import os.path as op                      # system functions

from ....interfaces import io as nio           # Data i/o
from ....interfaces import utility as util     # utility
from ....pipeline import engine as pe          # pypeline engine
from ....interfaces import fsl as fsl
from ....interfaces import freesurfer as fs    # freesurfer
from ....interfaces import mrtrix as mrtrix
from ....interfaces import mrtrix3 as mrtrix3
from ....interfaces import cmtk as cmtk
from ....interfaces import dipy as dipy
from ....algorithms import misc as misc
from ..connectivity.nx import create_networkx_pipeline, create_cmats_to_csv_pipeline
from ...misc.utils import select_aparc_annot


def create_connectivity_pipeline(name="connectivity", parcellation_name='scale500', n_tracks=50000):
    """Creates a pipeline that does the same connectivity processing as in the
    :ref:`example_dmri_connectivity_advanced` example script. Given a subject id (and completed Freesurfer reconstruction)
    diffusion-weighted image, b-values, and b-vectors, the workflow will return the subject's connectome
    as a Connectome File Format (CFF) file for use in Connectome Viewer (http://www.cmtk.org).

    Example
    -------

    >>> from nipype.workflows.dmri.mrtrix.connectivity_mapping import create_connectivity_pipeline
    >>> conmapper = create_connectivity_pipeline("nipype_conmap")
    >>> conmapper.inputs.inputnode.subjects_dir = '.'
    >>> conmapper.inputs.inputnode.subject_id = 'subj1'
    >>> conmapper.inputs.inputnode.dwi = 'data.nii.gz'
    >>> conmapper.inputs.inputnode.bvecs = 'bvecs'
    >>> conmapper.inputs.inputnode.bvals = 'bvals'
    >>> conmapper.run()                 # doctest: +SKIP

    Inputs::

        inputnode.subject_id
        inputnode.subjects_dir
        inputnode.dwi
        inputnode.bvecs
        inputnode.bvals
        inputnode.resolution_network_file

    Outputs::

        outputnode.connectome
        outputnode.cmatrix
        outputnode.networks
        outputnode.fa
        outputnode.struct
        outputnode.tracts
        outputnode.rois
        outputnode.filtered_tractography
        outputnode.tdi
        outputnode.nxstatscff
        outputnode.nxcsv
        outputnode.cmatrices_csv
        outputnode.mean_fiber_length
        outputnode.median_fiber_length
        outputnode.fiber_length_std
    """

    inputnode_within = pe.Node(util.IdentityInterface(fields=["subject_id",
                                                              "dwi",
                                                              "bvecs",
                                                              "bvals",
                                                              "subjects_dir",
                                                              "resolution_network_file"]),
                               name="inputnode_within")

    FreeSurferSource = pe.Node(interface=nio.FreeSurferSource(), name='fssource')
    FreeSurferSourceLH = pe.Node(interface=nio.FreeSurferSource(), name='fssourceLH')
    FreeSurferSourceLH.inputs.hemi = 'lh'

    FreeSurferSourceRH = pe.Node(interface=nio.FreeSurferSource(), name='fssourceRH')
    FreeSurferSourceRH.inputs.hemi = 'rh'

    """
    Creating the workflow's nodes
    =============================
    """

    """
    Conversion nodes
    ----------------
    """

    """
    A number of conversion operations are required to obtain NIFTI files from the FreesurferSource for each subject.
    Nodes are used to convert the following:
        * Original structural image to NIFTI
        * Pial, white, inflated, and spherical surfaces for both the left and right hemispheres are converted to GIFTI for visualization in ConnectomeViewer
        * Parcellated annotation files for the left and right hemispheres are also converted to GIFTI

    """

    mri_convert_Brain = pe.Node(interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'nii'
    mri_convert_T1 = mri_convert_Brain.clone(name='mri_convert_T1')
    mri_convert_ROI = mri_convert_Brain.clone('mri_convert_ROI')

    mris_convertLH = pe.Node(interface=fs.MRIsConvert(), name='mris_convertLH')
    mris_convertLH.inputs.out_datatype = 'gii'
    mris_convertRH = mris_convertLH.clone('mris_convertRH')
    mris_convertRHwhite = mris_convertLH.clone('mris_convertRHwhite')
    mris_convertLHwhite = mris_convertLH.clone('mris_convertLHwhite')
    mris_convertRHinflated = mris_convertLH.clone('mris_convertRHinflated')
    mris_convertLHinflated = mris_convertLH.clone('mris_convertLHinflated')
    mris_convertRHsphere = mris_convertLH.clone('mris_convertRHsphere')
    mris_convertLHsphere = mris_convertLH.clone('mris_convertLHsphere')
    mris_convertLHlabels = mris_convertLH.clone('mris_convertLHlabels')
    mris_convertRHlabels = mris_convertLH.clone('mris_convertRHlabels')

    """
    ACT
    """
    
    coregister_act = pe.Node(interface=fsl.FLIRT(dof=6), name = 'coregister_act')
    coregister_act.inputs.cost = ('normmi')
    act_fsl = pe.Node(interface=mrtrix3.ACTPrepareFSL(), name = 'act_fsl')
    applyxfm_act = pe.Node(interface=fsl.ApplyXfm4D(), name = 'applyxfm_act')
    makegmwmi = pe.Node(interface=mrtrix3.MakeGMWMI(), name='makegmwmi')
    
    """
    Diffusion processing nodes
    --------------------------
    """
    brainmask = pe.Node(interface=mrtrix3.BrainMask(),name='brainmask')
    responsesd = pe.Node(interface=mrtrix3.ResponseSD(),name='responsesd')
    estimatefod = pe.Node(interface=mrtrix3.EstimateFOD(),name='estimatefod')
    tractography = pe.Node(interface=mrtrix3.Tractography(),name='tractography')
    tractography.inputs.n_tracks = n_tracks


    tracks2prob = pe.Node(interface=mrtrix3.ComputeTDI(),name='tracks2prob')
    MRconvert_tracks2prob = pe.Node(interface=mrtrix.MRConvert(),name='MRconvert_tracks2prob')
    MRconvert_tracks2prob.inputs.extension = 'nii'

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(),name='tck2trk')
    trk2tdi = pe.Node(interface=dipy.TrackDensityMap(),name='trk2tdi')

    """
    Structural segmentation nodes
    -----------------------------
    """

    """
    The following node identifies the transformation between the diffusion-weighted
    image and the structural image. This transformation is then applied to the tracts
    so that they are in the same space as the regions of interest.
    """

    coregister = pe.Node(interface=fsl.FLIRT(dof=6), name = 'coregister')
    coregister.inputs.cost = ('normmi')

    """
    Parcellation is performed given the aparc+aseg image from Freesurfer.
    The CMTK Parcellation step subdivides these regions to return a higher-resolution parcellation scheme.
    The parcellation used here is entitled "scale500" and returns 1015 regions.
    """

    parcellate = pe.Node(interface=cmtk.Parcellate(), name="Parcellate")
    parcellate.inputs.parcellation_name = parcellation_name
    parcellate.inputs.dilation = True

    """
    The CreateMatrix interface takes in the remapped aparc+aseg image as well as the label dictionary and fiber tracts
    and outputs a number of different files. The most important of which is the connectivity network itself, which is stored
    as a 'gpickle' and can be loaded using Python's NetworkX package (see CreateMatrix docstring). Also outputted are various
    NumPy arrays containing detailed tract information, such as the start and endpoint regions, and statistics on the mean and
    standard deviation for the fiber length of each connection. These matrices can be used in the ConnectomeViewer to plot the
    specific tracts that connect between user-selected regions.

    Here we choose the Lausanne2008 parcellation scheme, since we are incorporating the CMTK parcellation step.
    """

    creatematrix = pe.Node(interface=cmtk.CreateMatrix(), name="CreateMatrix")
    creatematrix.inputs.count_region_intersections = True

    """
    Next we define the endpoint of this tutorial, which is the CFFConverter node, as well as a few nodes which use
    the Nipype Merge utility. These are useful for passing lists of the files we want packaged in our CFF file.
    The inspect.getfile command is used to package this script into the resulting CFF file, so that it is easy to
    look back at the processing parameters that were used.
    """

    CFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="CFFConverter")
    CFFConverter.inputs.script_files = op.abspath(inspect.getfile(inspect.currentframe()))
    giftiSurfaces = pe.Node(interface=util.Merge(8), name="GiftiSurfaces")
    giftiLabels = pe.Node(interface=util.Merge(2), name="GiftiLabels")
    niftiVolumes = pe.Node(interface=util.Merge(3), name="NiftiVolumes")
    fiberDataArrays = pe.Node(interface=util.Merge(4), name="FiberDataArrays")

    """
    We also create a node to calculate several network metrics on our resulting file, and another CFF converter
    which will be used to package these networks into a single file.
    """

    networkx = create_networkx_pipeline(name='networkx')
    cmats_to_csv = create_cmats_to_csv_pipeline(name='cmats_to_csv')
    nfibs_to_csv = pe.Node(interface=misc.Matlab2CSV(), name='nfibs_to_csv')
    merge_nfib_csvs = pe.Node(interface=misc.MergeCSVFiles(), name='merge_nfib_csvs')
    merge_nfib_csvs.inputs.extra_column_heading = 'Subject'
    merge_nfib_csvs.inputs.out_file = 'fibers.csv'
    NxStatsCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="NxStatsCFFConverter")
    NxStatsCFFConverter.inputs.script_files = op.abspath(inspect.getfile(inspect.currentframe()))

    """
    Connecting the workflow
    =======================
    Here we connect our processing pipeline.
    """


    """
    Connecting the inputs, FreeSurfer nodes, and conversions
    --------------------------------------------------------
    """

    mapping = pe.Workflow(name='mapping')

    """
    First, we connect the input node to the FreeSurfer input nodes.
    """

    mapping.connect([(inputnode_within, FreeSurferSource,[("subjects_dir","subjects_dir")])])
    mapping.connect([(inputnode_within, FreeSurferSource,[("subject_id","subject_id")])])

    mapping.connect([(inputnode_within, FreeSurferSourceLH,[("subjects_dir","subjects_dir")])])
    mapping.connect([(inputnode_within, FreeSurferSourceLH,[("subject_id","subject_id")])])

    mapping.connect([(inputnode_within, FreeSurferSourceRH,[("subjects_dir","subjects_dir")])])
    mapping.connect([(inputnode_within, FreeSurferSourceRH,[("subject_id","subject_id")])])

    mapping.connect([(inputnode_within, parcellate,[("subjects_dir","subjects_dir")])])
    mapping.connect([(inputnode_within, parcellate,[("subject_id","subject_id")])])
    mapping.connect([(parcellate, mri_convert_ROI,[('roiv_file','in_file')])])

    """
    Nifti conversion for subject's stripped brain image from Freesurfer:
    """

    mapping.connect([(FreeSurferSource, mri_convert_Brain,[('brain','in_file')])])
    mapping.connect([(FreeSurferSource, mri_convert_T1,[('T1','in_file')])])

    """
    Surface conversions to GIFTI (pial, white, inflated, and sphere for both hemispheres)
    """

    mapping.connect([(FreeSurferSourceLH, mris_convertLH,[('pial','in_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRH,[('pial','in_file')])])
    mapping.connect([(FreeSurferSourceLH, mris_convertLHwhite,[('white','in_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRHwhite,[('white','in_file')])])
    mapping.connect([(FreeSurferSourceLH, mris_convertLHinflated,[('inflated','in_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRHinflated,[('inflated','in_file')])])
    mapping.connect([(FreeSurferSourceLH, mris_convertLHsphere,[('sphere','in_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRHsphere,[('sphere','in_file')])])

    """
    The annotation files are converted using the pial surface as a map via the MRIsConvert interface.
    One of the functions defined earlier is used to select the lh.aparc.annot and rh.aparc.annot files
    specifically (rather than e.g. rh.aparc.a2009s.annot) from the output list given by the FreeSurferSource.
    """

    mapping.connect([(FreeSurferSourceLH, mris_convertLHlabels,[('pial','in_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRHlabels,[('pial','in_file')])])
    mapping.connect([(FreeSurferSourceLH, mris_convertLHlabels, [(('annot', select_aparc_annot), 'annot_file')])])
    mapping.connect([(FreeSurferSourceRH, mris_convertRHlabels, [(('annot', select_aparc_annot), 'annot_file')])])

    """
    Make the 5TT image for Anatomically Constrained Tractography
    ---------------------
    we coregister the 5tt image to the diffusion image and create am GM/WM interface mask
    """

    mapping.connect([(mri_convert_T1, act_fsl,[('out_file','in_file')])])
    
    mapping.connect([(act_fsl, coregister_act,[('out_file','in_file')])])
    mapping.connect([(inputnode_within, coregister_act,[("dwi","reference")])])
    
    mapping.connect([(act_fsl, applyxfm_act,[('out_file','in_file')])])
    mapping.connect([(coregister_act, applyxfm_act,[('out_matrix_file','in_matrix_file')])])
    mapping.connect([(inputnode_within, applyxfm_act,[("dwi","reference")])])
    mapping.connect([(applyxfm_act, makegmwmi,[("out_file","in_file")])])
    

    """
    Diffusion Processing
    --------------------
    Now we connect the tensor computations:
    """
    mapping.connect([(inputnode_within, brainmask,[("dwi","in_file")])])
    mapping.connect([(inputnode_within, brainmask,[("bvecs","in_bvec")])])
    mapping.connect([(inputnode_within, brainmask,[("bvals","in_bval")])])
    
    mapping.connect([(inputnode_within, responsesd,[("dwi","in_file")])])
    mapping.connect([(inputnode_within, responsesd,[("bvecs","in_bvec")])])
    mapping.connect([(inputnode_within, responsesd,[("bvals","in_bval")])])
    mapping.connect([(brainmask, responsesd,[("out_file","in_mask")])])
    
    mapping.connect([(inputnode_within, estimatefod,[("dwi","in_file")])])
    mapping.connect([(inputnode_within, estimatefod,[("bvecs","in_bvec")])])
    mapping.connect([(inputnode_within, estimatefod,[("bvals","in_bval")])])
    mapping.connect([(brainmask, estimatefod,[("out_file","in_mask")])])
    mapping.connect([(responsesd, estimatefod,[("out_file","response")])])
    
    mapping.connect([(estimatefod, tractography,[("out_file","in_file")])])
    #mapping.connect([(brainmask, tractography,[("out_file","seed_image")])])
    #mapping.connect([(brainmask, tractography,[("out_file","roi_mask")])])
    mapping.connect([(applyxfm_act, tractography,[("out_file","act_file")])])
    mapping.connect([(makegmwmi, tractography,[("out_file","seed_gmwmi")])])
    
    
    mapping.connect([(tractography, tracks2prob,[("out_file","in_file")])])
    mapping.connect([(inputnode_within, tracks2prob,[("dwi","reference")])])
    mapping.connect([(tracks2prob, MRconvert_tracks2prob,[("out_file","in_file")])])

    """
    Structural Processing
    ---------------------
    First, we coregister the diffusion image to the structural image
    """

    mapping.connect([(inputnode_within, coregister,[("dwi","in_file")])])
    mapping.connect([(mri_convert_Brain, coregister,[('out_file','reference')])])

    """
    The MRtrix-tracked fibers are converted to TrackVis format (with voxel and data dimensions grabbed from the DWI).
    The connectivity matrix is created with the transformed .trk fibers and the parcellation file.
    """

    mapping.connect([(inputnode_within, tck2trk,[("dwi","image_file")])])
    mapping.connect([(mri_convert_Brain, tck2trk,[("out_file","registration_image_file")])])
    mapping.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])
    mapping.connect([(tractography, tck2trk,[("out_file","in_file")])])
    mapping.connect([(tck2trk, creatematrix,[("out_file","tract_file")])])
    mapping.connect([(tck2trk, trk2tdi,[("out_file","in_file")])])
    mapping.connect(inputnode_within, 'resolution_network_file',
                    creatematrix, 'resolution_network_file')
    mapping.connect([(inputnode_within, creatematrix,[("subject_id","out_matrix_file")])])
    mapping.connect([(inputnode_within, creatematrix,[("subject_id","out_matrix_mat_file")])])
    mapping.connect([(parcellate, creatematrix,[("roiv_file","roi_file")])])

    """
    The merge nodes defined earlier are used here to create lists of the files which are
    destined for the CFFConverter.
    """

    mapping.connect([(mris_convertLH, giftiSurfaces,[("converted","in1")])])
    mapping.connect([(mris_convertRH, giftiSurfaces,[("converted","in2")])])
    mapping.connect([(mris_convertLHwhite, giftiSurfaces,[("converted","in3")])])
    mapping.connect([(mris_convertRHwhite, giftiSurfaces,[("converted","in4")])])
    mapping.connect([(mris_convertLHinflated, giftiSurfaces,[("converted","in5")])])
    mapping.connect([(mris_convertRHinflated, giftiSurfaces,[("converted","in6")])])
    mapping.connect([(mris_convertLHsphere, giftiSurfaces,[("converted","in7")])])
    mapping.connect([(mris_convertRHsphere, giftiSurfaces,[("converted","in8")])])

    mapping.connect([(mris_convertLHlabels, giftiLabels,[("converted","in1")])])
    mapping.connect([(mris_convertRHlabels, giftiLabels,[("converted","in2")])])

    mapping.connect([(parcellate, niftiVolumes,[("roiv_file","in1")])])
    mapping.connect([(inputnode_within, niftiVolumes,[("dwi","in2")])])
    mapping.connect([(mri_convert_Brain, niftiVolumes,[("out_file","in3")])])

    mapping.connect([(creatematrix, fiberDataArrays,[("endpoint_file","in1")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("endpoint_file_mm","in2")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("fiber_length_file","in3")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("fiber_label_file","in4")])])

    """
    This block actually connects the merged lists to the CFF converter. We pass the surfaces
    and volumes that are to be included, as well as the tracts and the network itself. The currently
    running pipeline (dmri_connectivity_advanced.py) is also scraped and included in the CFF file. This
    makes it easy for the user to examine the entire processing pathway used to generate the end
    product.
    """

    mapping.connect([(giftiSurfaces, CFFConverter,[("out","gifti_surfaces")])])
    mapping.connect([(giftiLabels, CFFConverter,[("out","gifti_labels")])])
    mapping.connect([(creatematrix, CFFConverter,[("matrix_files","gpickled_networks")])])
    mapping.connect([(niftiVolumes, CFFConverter,[("out","nifti_volumes")])])
    mapping.connect([(fiberDataArrays, CFFConverter,[("out","data_files")])])
    mapping.connect([(creatematrix, CFFConverter,[("filtered_tractography","tract_files")])])
    mapping.connect([(inputnode_within, CFFConverter,[("subject_id","title")])])

    """
    The graph theoretical metrics which have been generated are placed into another CFF file.
    """

    mapping.connect([(inputnode_within, networkx,[("subject_id","inputnode.extra_field")])])
    mapping.connect([(creatematrix, networkx,[("intersection_matrix_file","inputnode.network_file")])])

    mapping.connect([(networkx, NxStatsCFFConverter,[("outputnode.network_files","gpickled_networks")])])
    mapping.connect([(giftiSurfaces, NxStatsCFFConverter,[("out","gifti_surfaces")])])
    mapping.connect([(giftiLabels, NxStatsCFFConverter,[("out","gifti_labels")])])
    mapping.connect([(niftiVolumes, NxStatsCFFConverter,[("out","nifti_volumes")])])
    mapping.connect([(fiberDataArrays, NxStatsCFFConverter,[("out","data_files")])])
    mapping.connect([(inputnode_within, NxStatsCFFConverter,[("subject_id","title")])])

    mapping.connect([(inputnode_within, cmats_to_csv,[("subject_id","inputnode.extra_field")])])
    mapping.connect([(creatematrix, cmats_to_csv,[("matlab_matrix_files","inputnode.matlab_matrix_files")])])
    mapping.connect([(creatematrix, nfibs_to_csv,[("stats_file","in_file")])])
    mapping.connect([(nfibs_to_csv, merge_nfib_csvs,[("csv_files","in_files")])])
    mapping.connect([(inputnode_within, merge_nfib_csvs,[("subject_id","extra_field")])])


    """
    Create a higher-level workflow
    --------------------------------------
    Finally, we create another higher-level workflow to connect our mapping workflow with the info and datagrabbing nodes
    declared at the beginning. Our tutorial can is now extensible to any arbitrary number of subjects by simply adding
    their names to the subject list and their data to the proper folders.
    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "dwi", "bvecs", "bvals", "subjects_dir", "resolution_network_file"]), name="inputnode")

    outputnode = pe.Node(interface = util.IdentityInterface(fields=["struct",
                                                                "tracts",
                                                                "connectome",
                                                                "nxstatscff",
                                                                "nxmatlab",
                                                                "nxcsv",
                                                                "fiber_csv",
                                                                "cmatrices_csv",
                                                                "nxmergedcsv",
                                                                "cmatrix",
                                                                "networks",
                                                                "filtered_tracts",
                                                                "rois",
                                                                "dipy_tdi","mrtrix_tdi",
                                                                "mean_fiber_length",
                                                                "median_fiber_length",
                                                                "fiber_length_std"]),
                                        name="outputnode")

    connectivity = pe.Workflow(name="connectivity")
    connectivity.base_output_dir=name
    connectivity.base_dir=name

    connectivity.connect([(inputnode, mapping, [("dwi", "inputnode_within.dwi"),
                                              ("bvals", "inputnode_within.bvals"),
                                              ("bvecs", "inputnode_within.bvecs"),
                                              ("subject_id", "inputnode_within.subject_id"),
                                              ("subjects_dir", "inputnode_within.subjects_dir"),
                                              ("resolution_network_file", "inputnode_within.resolution_network_file")])
                                              ])

    connectivity.connect([(mapping, outputnode, [("tck2trk.out_file", "tracts"),
                                                ("CFFConverter.connectome_file", "connectome"),
                                                ("NxStatsCFFConverter.connectome_file", "nxstatscff"),
                                                ("CreateMatrix.matrix_mat_file", "cmatrix"),
                                                ("CreateMatrix.mean_fiber_length_matrix_mat_file", "mean_fiber_length"),
                                                ("CreateMatrix.median_fiber_length_matrix_mat_file", "median_fiber_length"),
                                                ("CreateMatrix.fiber_length_std_matrix_mat_file", "fiber_length_std"),
                                                ("CreateMatrix.matrix_files", "networks"),
                                                ("CreateMatrix.filtered_tractographies", "filtered_tracts"),
                                                ("merge_nfib_csvs.csv_file", "fiber_csv"),
                                                ("mri_convert_ROI.out_file", "rois"),
                                                ("trk2tdi.out_file", "dipy_tdi"),
                                                ("mri_convert_Brain.out_file", "struct"),
                                                ("MRconvert_tracks2prob.converted", "mrtrix_tdi")])
        ])

    connectivity.connect([(cmats_to_csv, outputnode,[("outputnode.csv_file","cmatrices_csv")])])
    connectivity.connect([(networkx, outputnode,[("outputnode.csv_files","nxcsv")])])
    return connectivity
