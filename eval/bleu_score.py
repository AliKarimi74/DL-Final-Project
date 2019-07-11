import sys, argparse
import nltk


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate text edit distance.')

    parser.add_argument('--target-formulas', dest='target_file',
                        type=str, required=True,
                        help=(
                            'target formulas file'
                        ))

    parser.add_argument('--predicted-formulas', dest='predicted_file',
                        type=str, required=True,
                        help=(
                            'predicted formulas file'
                        ))

    parser.add_argument('--ngram', dest='ngram',
                        type=int, required=True,
                        help=(
                            'predicted formulas file'
                        ))

    parameters = parser.parse_args(args)
    return parameters


def bleu_eval(target_formulas, predicted_formulas, ngram=5):
    total_bleu_score = 0
    n = len(target_formulas)
    if len(target_formulas) != len(predicted_formulas):
        print("number of formulas doesn't match")
        return

    for tf, pf in zip(target_formulas, predicted_formulas):
        tf_ = tf.strip()
        pf_ = pf.strip()
        true_token = tf_.split(' ')
        predicted_tokens = pf_.split(' ')
        l = min(ngram, len(true_token), len(predicted_tokens))
        if l == 0:
            print("Encountered formula with zero length")
        bleu_score = nltk.translate.bleu_score.sentence_bleu([true_token], predicted_tokens,
                                                             weights=[1.0 / l for _ in range(l)])
        total_bleu_score += bleu_score
    score = float(total_bleu_score) / n
    print('BLEU Score: %f' % score)
    return score


def main(args):
    parameters = process_args(args)

    target_formulas_file = parameters.target_file
    predicted_formulas_file = parameters.predicted_file
    ngram = parameters.ngram

    target_formulas = open(target_formulas_file).readlines()
    predicted_formulas = open(predicted_formulas_file).readlines()

    bleu_eval(target_formulas, predicted_formulas, ngram)


if __name__ == '__main__':
    main(sys.argv[1:])

