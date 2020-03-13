from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'test_instance',
        'cpp_constructor_args',
        'has_parity',
        'device',
        'cpp_output_tmp_folder',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])
