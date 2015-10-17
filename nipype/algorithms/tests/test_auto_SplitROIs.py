# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..misc import SplitROIs

def test_SplitROIs_inputs():
    input_map = dict(in_file=dict(mandatory=True,
    ),
    in_mask=dict(),
    roi_size=dict(),
    )
    inputs = SplitROIs.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_SplitROIs_outputs():
    output_map = dict(out_files=dict(),
    out_index=dict(),
    out_masks=dict(),
    )
    outputs = SplitROIs.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

