from sample_augment.core import step, Experiment, Config


@step
def with_optional(train_baseline: bool = True):
    assert train_baseline == False


def test_this():
    experiment = Experiment(config=Config.parse_obj({'train_baseline': False}))
    experiment.run('with_optional')
