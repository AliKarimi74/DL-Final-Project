import random
import numpy as np

from data.data_generator import DataGenerator
from utils.logger import log as LOG
from .bleu_score import bleu_eval
from .edit_distance import edit_distance_eval


def evaluation(session, model, mode='validation', percent_limit=None, save_path=None):
    def log(msg, add_trailing=True):
        LOG(msg, add_trailing)
    dataset = DataGenerator(mode)

    target_formulas = []
    predicted_formulas = []
    pp_hist = []
    last_log_percentage = 0
    log_percentage_every = 10
    for epoch, percentage, images, formulas, _ in dataset.generator(1, percent_limit):
        target = dataset.decode_formulas(formulas)
        prediction, pp = model.predict(sess=session, images=images)
        prediction = dataset.decode_formulas(prediction)
        target_formulas += target
        predicted_formulas += prediction
        pp_hist += [pp]

        max_per = 1 if percent_limit is None else percent_limit
        percentage = int(100 * (percentage/max_per))
        if percentage >= last_log_percentage + log_percentage_every:
            idx = random.randint(0, len(prediction) - 1)
            last_log_percentage += log_percentage_every
            log('Evaluation prediction progress completion: {}%\ntrue -> {}\npred -> {}'.
                format(percentage, '' if len(target) <= idx else target[idx], prediction[idx]))

    if save_path is not None:
        with open(save_path, 'w+') as f:
            f.write('\n'.join(predicted_formulas))

    if len(target_formulas) != len(predicted_formulas):
        log("number of formulas doesn't match", False)
        return None

    bleu_score = bleu_eval(target_formulas, predicted_formulas)
    edit_distance_score = edit_distance_eval(target_formulas, predicted_formulas)
    pp_mean = np.mean(np.array(pp_hist))
    log('Bleu score:            {0:2.3f} %'.format(100 * bleu_score))
    log('Edit distance score:   {0:2.3f} %'.format(100 * edit_distance_score))
    log('Perplexity:            {0:10.3f} %'.format(pp_mean))

    return bleu_score, edit_distance_score
