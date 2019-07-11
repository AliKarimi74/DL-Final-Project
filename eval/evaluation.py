import random

from data_generator import DataGenerator
from .bleu_score import bleu_eval
from .edit_distance import edit_distance_eval


def evaluation(session, model, mode='validation'):
    dataset = DataGenerator(mode)

    target_formulas = []
    predicted_formulas = []
    last_log_percentage = 0
    log_percentage_every = 10
    for epoch, percentage, images, formulas, _ in dataset.generator(1):
        target = dataset.decode_formulas(formulas)
        prediction = model.predict(sess=session, images=images)[0]
        prediction = dataset.decode_formulas(prediction)
        target_formulas += target
        predicted_formulas += prediction

        percentage = int(100 * (percentage + 0.1))
        if percentage >= last_log_percentage + log_percentage_every:
            idx = random.randint(0, len(prediction))
            last_log_percentage += log_percentage_every
            print('Evaluation prediction progress completion: {}%\ntrue -> {}\npred -> {}'.
                  format(percentage, target[idx], prediction[idx]))

    if len(target_formulas) != len(predicted_formulas):
        print("number of formulas doesn't match")
        return

    exact_match = 0
    for tf, pf in zip(target_formulas, predicted_formulas):
        tf, pf = tf.strip(), pf.strip()
        if tf == pf:
            exact_match += 1

    exact_match_score = float(exact_match) / len(target_formulas)
    print('Exact match: {0:2.3f} %'.format(100 * exact_match_score))

    bleu_score = bleu_eval(target_formulas, predicted_formulas)
    edit_distance_score = edit_distance_eval(target_formulas, predicted_formulas)

    return exact_match_score, bleu_score, edit_distance_score
