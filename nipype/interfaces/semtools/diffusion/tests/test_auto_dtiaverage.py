# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..diffusion import dtiaverage

def test_dtiaverage_inputs():
    input_map = dict(DTI_double=dict(argstr='--DTI_double ',
    ),
    args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputs=dict(argstr='--inputs %s...',
    ),
    tensor_output=dict(argstr='--tensor_output %s',
    hash_files=False,
    ),
    terminal_output=dict(nohash=True,
    ),
    verbose=dict(argstr='--verbose ',
    ),
    )
    inputs = dtiaverage.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_dtiaverage_outputs():
    output_map = dict(tensor_output=dict(),
    )
    outputs = dtiaverage.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

