import random

from data_generator import DataGenerator
from .bleu_score import bleu_eval
from .edit_distance import edit_distance_eval


def evaluation(session, model, mode='validation', percent_limit=None):
    def log(msg):
        return msg + '\n' + ('-'*150)
    dataset = DataGenerator(mode)

    target_formulas = []
    predicted_formulas = []
    last_log_percentage = 0
    log_percentage_every = 10
    for epoch, percentage, images, formulas, _ in dataset.generator(1, percent_limit):
        target = dataset.decode_formulas(formulas)
        prediction = model.predict(sess=session, images=images)[0]
        prediction = dataset.decode_formulas(prediction)
        target_formulas += target
        predicted_formulas += prediction

        max_per = 1 if percent_limit is None else percent_limit
        percentage = int(100 * (percentage/max_per + 0.1))
        if percentage >= last_log_percentage + log_percentage_every:
            idx = random.randint(0, len(prediction) - 1)
            last_log_percentage += log_percentage_every
            print(log('Evaluation prediction progress completion: {}%\ntrue -> {}\npred -> {}').
                  format(percentage, target[idx], prediction[idx]))

    if len(target_formulas) != len(predicted_formulas):
        print("number of formulas doesn't match")
        return

    exact_match = 0
    exact_match_log_limit = 10
    for tf, pf in zip(target_formulas, predicted_formulas):
        tf, pf = tf.strip(), pf.strip()
        if tf == pf:
            exact_match += 1
            if exact_match <= exact_match_log_limit:
                print('         Exact match sample:\ntrue -> {}\npred -> {}'.format(tf, pf))

    exact_match_score = float(exact_match) / len(target_formulas)
    print(log('Exact match: {0:2.3f} %').format(100 * exact_match_score))

    bleu_score = bleu_eval(target_formulas, predicted_formulas)
    print(log('')[1:])
    edit_distance_score = edit_distance_eval(target_formulas, predicted_formulas)
    print(log('')[1:])

    return exact_match_score, bleu_score, edit_distance_score
