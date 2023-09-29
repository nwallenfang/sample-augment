from sample_augment.core import step
from sample_augment.data.train_test_split import TrainSet
from sample_augment.utils import log


@step
def get_training_size(training: TrainSet):
    log.info(training.label_tensor.shape)
