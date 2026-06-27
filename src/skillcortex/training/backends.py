from ..backends.legacy import ExecutionFixture, evaluator_backend, model_backend, trainer_backend


aggregate_results = evaluator_backend.aggregate_results
extract_code = evaluator_backend.extract_code
fuzzy_match = evaluator_backend.fuzzy_match
generate_text = model_backend.generate_text
load_model = model_backend.load_model
python_syntax_valid = evaluator_backend.python_syntax_valid
research_metadata = trainer_backend.training_metadata
run_fixture = evaluator_backend.run_fixture
saved_parameter_count = trainer_backend.saved_parameter_count
training_config = trainer_backend.training_config
training_command = trainer_backend.build_generic_training_command
