import tempfile
import shutil
from string import Template
import unittest
from torch.testing._internal.common_cuda import TEST_CUDA

import torch
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.utils import TorchNNTestParams

# Check 1: Module implementation correctness check:

# Step 1: Translate ctor args from Python layer to C++ layer
# Step 2: Construct a C++ layer, run forward and backward on it, save all its params/buffers/gradients into a ScriptModule
# Step 3: Load that ScriptModule into Python, and compare output/params/buffers/gradients with Python layer (forward and backward)

# yf225 TODO: move to common utils?
devices = ['cpu', 'cuda']

# yf225 TODO: move to common utils?
TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
#include <torch/script.h>
"""

TORCH_NN_MODULE_TEST_FORWARD = Template("""\n
void ${module_variant_name}_test_forward(
    const std::string& saved_module_path,
    const std::string& device,
    ${input_arg_declarations}) {
  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});

  torch::jit::save(
    "${cpp_output_tmp_folder}_${module_variant_name}_test_forward.pt",
    torch::IValue(cpp_output));

  // yf225 TODO: can we save output as IValue into a file, and then do the comparison in Python?
  //  TORCH_CHECK(
  //  check_tensor_equality(cpp_output, python_output),
  //  GENERATE_PARITY_TEST_ERROR_MSG(
  //    "forward output",
  //    cpp_output,
  //    python_output));
}
""")

TORCH_NN_MODULE_TEST_BACKWARD = Template("""\n
void ${module_variant_name}_test_backward(
    const std::string& saved_module_path,
    const std::string& device,
    ${input_arg_declarations}) {
  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});
  cpp_output.sum().backward();

  // yf225 TODO: can we save all the gradients into a ScriptModule into a file, and then do the comparison in Python?
  // for (size_t i = 0; i < module->parameters().size(); i++) {
  //   auto named_param = module->named_parameters()[i];
  //   auto grad = python_grad_module->parameters()[i];
  //   TORCH_CHECK(
  //     check_tensor_equality(named_param->grad(), grad),
  //     GENERATE_PARITY_TEST_ERROR_MSG(
  //       "gradient of `" + named_param.key() + "`",
  //       named_param->grad(),
  //       grad));
  // }
}
""")

def _test_torch_nn_module_variant(test_params):
    def get_python_ignored_attrs(module_metadata):
      return list(TORCH_NN_MODULE_IGNORED_ATTRS) + module_metadata.python_ignored_attrs

    # yf225 TODO: move to common utils?
    # yf225 TODO: we should check in a copy of the generated source code, and then run consistency test (compare old vs. newly generated)
    def generate_test_cpp_sources(test_params, template):
      input_args = self._get_forward_input_args(test_params)
      input_arg_types = [self._python_arg_to_cpp_arg(arg).type for arg in list(input_args)]
      input_args = ['arg{}'.format(str(i)) for i in range(len(input_arg_types))]
      input_arg_declarations = ['{} {}'.format(arg_type, arg_name) for arg_type, arg_name in zip(input_arg_types, input_args)]
      test_cpp_sources = template.substitute(
        module_variant_name=test_params.module_variant_name,
        module_qualified_name='torch::nn::{}'.format(test_params.module_name),
        cpp_constructor_args=test_params.cpp_constructor_args,
        input_arg_declarations=',\n'.join(input_arg_declarations),
        input_args=',\n'.join(input_args),
        cpp_output_tmp_folder=test_params.cpp_output_tmp_folder,
      )
      print(test_cpp_sources)
      return test_cpp_sources

    def setup_forward_test(test_params):
      device = test_params.device
      python_constructor = test_params.test_instance.constructor
      python_constructor_args = test_params.test_instance.constructor_args
      input_args = self._get_forward_input_args(test_params)

      torch.manual_seed(2)
      module = python_constructor(*python_constructor_args).to(device)
      python_output = module(*input_args)

      return (([module], device, python_output, input_args),
          generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD))

    def setup_backward_test(test_params):
      device = test_params.device
      python_constructor = test_params.test_instance.constructor
      python_constructor_args = test_params.test_instance.constructor_args
      input_args = self._get_forward_input_args(test_params)

      torch.manual_seed(2)
      module = python_constructor(*python_constructor_args).to(device)
      python_output = module(*input_args)
      python_output.sum().backward()
      # JIT tracing does not save a module's parameters' gradients into ScriptModule.
      # Instead, we create another module `grad_module` with the same structure as `module`,
      # and use `grad_module`'s parameters to save `module`'s corresponding parameters'
      # gradients. Then, we trace both `module` and `grad_module`, serialize them and
      # pass them into C++ for parity testing.
      grad_module = copy.deepcopy(module)
      for param_name, param_value in module.named_parameters():
        if param_value.grad is not None:
          grad_module[param_name] = param_value.grad

      return (([module, grad_module], device, input_args),
          generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_BACKWARD))

    def serialize_module_into_file(script_module):
      module_file = tempfile.NamedTemporaryFile(delete=False)
      script_module.save(module_file.name)
      module_file.close()
      return module_file.name

    def test_basic_methods(test_params):
      module_metadata = torch_nn_modules.module_metadata_map[test_params.module_name]
      module_variant_name = test_params.module_variant_name
      input_args = self._get_forward_input_args(test_params)

      args_map = {}

      cpp_sources = TORCH_NN_MODULE_COMMON_TEST_HARNESS + module_metadata.cpp_sources

      torch_nn_test_methods = [
        ('test_forward', setup_forward_test),
        # ('test_backward', setup_backward_test),
      ]
      for method_name, setup_test in torch_nn_test_methods:
        args_map[method_name], test_cpp_sources = setup_test(test_params)
        cpp_sources += test_cpp_sources

      cpp_module = self._compile_cpp_code_inline(
        name=test_params.module_variant_name,
        cpp_sources=cpp_sources,
        functions=['{}_test_{}'.format(
          test_params.module_variant_name,
          method_name) for method_name, _ in torch_nn_test_methods])

      for method_name, _ in torch_nn_test_methods:
        args = args_map[method_name]
        modules = args[0]
        script_modules = [torch.jit.trace(module, input_args) for module in modules]
        module_file_names = [serialize_module_into_file(script_module) for script_module in script_modules]

        cpp_args = module_file_names[:]
        for arg in args[1:]:
          if isinstance(arg, tuple):
            cpp_args += list(arg)
          elif isinstance(arg, list):
            cpp_args += arg
          else:
            cpp_args.append(arg)

        try:
          cpp_test_name = '{}_test_{}'.format(module_variant_name, method_name)
          cpp_test_fn = getattr(cpp_module, cpp_test_name)
          if not test_params.has_parity:
            with self.assertRaisesRegex(RuntimeError, "Parity test failed"):
              cpp_test_fn(*cpp_args)
          else:
            cpp_test_fn(*cpp_args)
            ivalue = torch.jit.load("{}_{}_{}.pt", test_params.cpp_output_tmp_folder, test_params.module_variant_name, method_name)
            print(ivalue)
        finally:
          # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
          # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
          # we close the file after creation and try to remove it manually.
          for module_file_name in module_file_names:
            try:
              os.remove(module_file_name)
            except OSError as e:
              warnings.warn("Unable to remove {}, got error: {}".format(module_file_name, str(e)))

      # Remove temporary folder that stores C++ outputs
      shutil.rmtree(test_params.cpp_output_tmp_folder)

    test_basic_methods(test_params)

# yf225 TODO: move to common utils?
def _compute_module_name(test_params_dict):
    fullname = test_params_dict.get('fullname', None)
    if fullname:
        # NOTE: This doesn't work for some of the `wrap_functional` module tests such as "interpolate_nearest_1d",
        # because in that case the module `interpolate` is not in `torch.nn` but rather in `torch.nn.functional`.
        # We will fix this when we have parity tests for `torch.nn.functional` modules.
        module_name = fullname.split('_')[0]
    else:
        module_name = test_params_dict.get('module_name')
    return module_name

# yf225 TODO: move to common utils?
def _process_test_params_for_module(test_params_dict, module_metadata, device, is_criterion):
    module_name = _compute_module_name(test_params_dict)
    test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
    if is_criterion:
        test = common_nn.CriterionTest(**test_params_dict)
    else:
        test = common_nn.ModuleTest(**test_params_dict)
    # yf225 TODO: can we remove the magic number `5` here?
    module_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')    

    return TorchNNTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test,
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args'),
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
        cpp_output_tmp_folder=tempfile.mkdtemp(),
    )

# yf225 TODO: move to common utils?
def has_test(unit_test_class, test_name):
    return hasattr(unit_test_class, test_name)

# yf225 TODO: move to common utils?
def add_test(unit_test_class, test_name, test_fn):
    if has_test(unit_test_class, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(unit_test_class, test_name, test_fn)

def add_torch_nn_module_impl_parity_tests(parity_table, unit_test_class, torch_nn_modules, module_tests, is_criterion):
  torch_nn_test_params_map = {}
  for test_params_dict in module_tests:
    # Skip all `torch.nn.functional` tests, since they are handled by another test suite.
    if 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
      continue

    module_name = _compute_module_name(test_params_dict)

    assert hasattr(torch.nn, module_name), \
      "`torch.nn` doesn't have module `{}`. ".format(module_name) + \
      "If you are adding a new test, please set `fullname` using format `ModuleName_desc`, " + \
      "or set `module_name` using format `ModuleName`."

    module_full_name = 'torch::nn::' + module_name
    # If equivalent module in C++ frontend doesn't exist, we don't do the parity test.
    if module_full_name not in parity_table['torch::nn']:
      continue

    has_impl_parity, _ = parity_table['torch::nn'][module_full_name]

    def add_variant_test_for_module(module_name, test_params_dict, has_impl_parity, torch_nn_modules):
      module_metadata = torch_nn_modules.module_metadata_map[module_name]
      for device in devices:
        test_params = _process_test_params_for_module(
          test_params_dict=test_params_dict,
          module_metadata=module_metadata,
          device=device,
          is_criterion=is_criterion)
        test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
        torch_nn_test_params_map[test_name] = test_params

        def test_fn(self):
          self._test_torch_nn_module_variant(test_params=torch_nn_test_params_map[self._testMethodName])

        if device == 'cuda':
          test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

        # If `Implementation Parity` entry in parity table for this module is `No`,
        # we mark the test as expected failure.
        if not has_impl_parity:
          test_fn = unittest.expectedFailure(test_fn)

        add_test(unit_test_class, test_name, test_fn)

    add_variant_test_for_module(
      module_name=module_name,
      test_params_dict=test_params_dict,
      has_impl_parity=has_impl_parity,
      torch_nn_modules=torch_nn_modules)

def add_tests(unit_test_class, module_tests, criterion_tests, torch_nn_modules, parity_table):
  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=module_tests,
    is_criterion=False)

  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=criterion_tests,
    is_criterion=True)
