# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:51:18 2015

@author: brain
"""

import inspect
import os.path as op                      # system functions

from ....interfaces import utility as util     # utility
from ....pipeline import engine as pe          # pypeline engine
from ....interfaces import (fsl, ants, cmtk, dipy, mrtrix, mrtrix3)
from ....algorithms import misc
from ..connectivity.nx import create_networkx_pipeline, create_cmats_to_csv_pipeline


def create_connectivity_pipeline(name="connectivity", n_tracks=50000):
    """
    """

    inputnode_within = pe.Node(util.IdentityInterface(fields=["subject_id",
                                                              "dwi",
                                                              "bvecs",
                                                              "bvals",
                                                              "t1",
                                                              "resolution_network_file",
                                                              "roi_file",
                                                              "template_file"]),
                                                      name="inputnode_within")


    """
    Creating the workflow's nodes
    =============================
    """

    """
    Conversion nodes
    ----------------
    """
    
    
    fiberDataArrays = pe.Node(interface=util.Merge(4), name="FiberDataArrays")

    """
    ACT
    """
    
    coregister_act = pe.Node(interface=fsl.FLIRT(dof=6), name = 'coregister_act')
    coregister_act.inputs.cost = ('normmi')
    act_fsl = pe.Node(interface=mrtrix3.ACTPrepareFSL(), name = 'act_fsl')
    applyxfm_act = pe.Node(interface=fsl.ApplyXfm4D(), name = 'applyxfm_act')
    makegmwmi = pe.Node(interface=mrtrix3.MakeGMWMI(), name = 'makegmwmi')
    
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
    We register the template to the anatomy, then use the resulting 
    transformation to warp the roi file to the anatomy 
    """
    
    reg = pe.Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2 + [[100, 50, 30]])
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.float = True
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    
    warproi = pe.Node(ants.ApplyTransforms(), name='warproi')
    warproi.inputs.input_image_type = 3
    warproi.inputs.interpolation = 'Linear'
    warproi.inputs.invert_transform_flags = [False]
    warproi.inputs.terminal_output = 'file'
    warproi.inputs.args = '--float'
    warproi.inputs.num_threads = 4

    

    """
    The following node identifies the transformation between the diffusion-weighted
    image and the structural image. This transformation is then applied to the tracts
    so that they are in the same space as the regions of interest.
    """

    coregister = pe.Node(interface=fsl.FLIRT(dof=6), name = 'coregister')
    coregister.inputs.cost = ('normmi')

    """
    The CreateMatrix interface takes in the remapped aparc+aseg image as well as the label dictionary and fiber tracts
    and outputs a number of different files. The most important of which is the connectivity network itself, which is stored
    as a 'gpickle' and can be loaded using Python's NetworkX package (see CreateMatrix docstring). Also outputted are various
    NumPy arrays containing detailed tract information, such as the start and endpoint regions, and statistics on the mean and
    standard deviation for the fiber length of each connection. These matrices can be used in the ConnectomeViewer to plot the
    specific tracts that connect between user-selected regions.

    """

    creatematrix = pe.Node(interface=cmtk.CreateMatrix(), name="CreateMatrix")
    creatematrix.inputs.count_region_intersections = True


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
    Make the 5TT image for Anatomically Constrained Tractography
    ---------------------
    we coregister the 5tt image to the diffusion image and create am GM/WM interface mask
    """

    mapping.connect([(inputnode_within, act_fsl,[('t1','in_file')])])
    
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
    we coregister the roi image to the structural image
    we coregister the diffusion image to the structural image
    """   
    
    mapping.connect(inputnode_within, 'template_file', reg, 'moving_image')
    mapping.connect(act_fsl,'out_file', reg,'fixed_image')
    
    mapping.connect(act_fsl,'out_file', warproi,'reference_image')
    mapping.connect(inputnode_within, 'roi_file', warproi, 'input_image')
    mapping.connect(reg, 'composite_transform', warproi, 'transforms')

    mapping.connect([(inputnode_within, coregister,[("dwi","in_file")])])
    mapping.connect([(act_fsl, coregister,[('out_file','reference')])])

    """
    The MRtrix-tracked fibers are converted to TrackVis format (with voxel and data dimensions grabbed from the DWI).
    The connectivity matrix is created with the transformed .trk fibers and the parcellation file.
    """

    mapping.connect([(inputnode_within, tck2trk,[("dwi","image_file")])])
    mapping.connect([(act_fsl, tck2trk,[("out_file","registration_image_file")])])
    mapping.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])
    mapping.connect([(tractography, tck2trk,[("out_file","in_file")])])
    mapping.connect([(tck2trk, creatematrix,[("out_file","tract_file")])])
    mapping.connect([(tck2trk, trk2tdi,[("out_file","in_file")])])
    mapping.connect(inputnode_within, 'resolution_network_file', creatematrix, 'resolution_network_file')
    mapping.connect([(inputnode_within, creatematrix,[("subject_id","out_matrix_file")])])
    mapping.connect([(inputnode_within, creatematrix,[("subject_id","out_matrix_mat_file")])])
    mapping.connect([(warproi, creatematrix,[("output_image","roi_file")])])

    """
    The merge nodes defined earlier are used here to create lists of the files which are
    destined for the CFFConverter.
    """


    mapping.connect([(creatematrix, fiberDataArrays,[("endpoint_file","in1")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("endpoint_file_mm","in2")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("fiber_length_file","in3")])])
    mapping.connect([(creatematrix, fiberDataArrays,[("fiber_label_file","in4")])])


    mapping.connect([(inputnode_within, networkx,[("subject_id","inputnode.extra_field")])])
    mapping.connect([(creatematrix, networkx,[("intersection_matrix_file","inputnode.network_file")])])

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

    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", 
                                                                 "dwi", 
                                                                 "bvecs", 
                                                                 "bvals", 
                                                                 "t1", 
                                                                 "resolution_network_file", 
                                                                 "roi_file",
                                                                 "template_file"
                                                                 ]), name="inputnode")

    outputnode = pe.Node(interface = util.IdentityInterface(fields=["struct",
                                                                "tracts",
                                                                "nxmatlab",
                                                                "nxcsv",
                                                                "fiber_csv",
                                                                "cmatrices_csv",
                                                                "nxmergedcsv",
                                                                "cmatrix",
                                                                "networks",
                                                                "filtered_tracts",
                                                                "dipy_tdi","mrtrix_tdi",
                                                                "mean_fiber_length",
                                                                "median_fiber_length",
                                                                "fiber_length_std"
                                                                ]), name="outputnode")

    connectivity = pe.Workflow(name="connectivity")
    connectivity.base_output_dir=name
    connectivity.base_dir=name

    connectivity.connect([(inputnode, mapping, [("dwi", "inputnode_within.dwi"),
                                              ("bvals", "inputnode_within.bvals"),
                                              ("bvecs", "inputnode_within.bvecs"),
                                              ("subject_id", "inputnode_within.subject_id"),
                                              ("t1", "inputnode_within.t1"),
                                              ("resolution_network_file", "inputnode_within.resolution_network_file"),
                                              ("roi_file", "inputnode_within.roi_file"),
                                              ("template_file", "inputnode_within.template_file")])
                                              ])

    connectivity.connect([(mapping, outputnode, [("tck2trk.out_file", "tracts"),
                                                ("CreateMatrix.matrix_mat_file", "cmatrix"),
                                                ("CreateMatrix.mean_fiber_length_matrix_mat_file", "mean_fiber_length"),
                                                ("CreateMatrix.median_fiber_length_matrix_mat_file", "median_fiber_length"),
                                                ("CreateMatrix.fiber_length_std_matrix_mat_file", "fiber_length_std"),
                                                ("CreateMatrix.matrix_files", "networks"),
                                                ("CreateMatrix.filtered_tractographies", "filtered_tracts"),
                                                ("merge_nfib_csvs.csv_file", "fiber_csv"),
                                                ("trk2tdi.out_file", "dipy_tdi"),
                                                ("act_fsl.out_file", "struct"),
                                                ("MRconvert_tracks2prob.converted", "mrtrix_tdi")])
        ])

    connectivity.connect([(cmats_to_csv, outputnode,[("outputnode.csv_file","cmatrices_csv")])])
    connectivity.connect([(networkx, outputnode,[("outputnode.csv_files","nxcsv")])])
    return connectivity
