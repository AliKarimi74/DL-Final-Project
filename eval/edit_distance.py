import sys, argparse
import distance


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

    parameters = parser.parse_args(args)
    return parameters


def edit_distance_eval(target_formulas, predicted_formulas):
    total_len = 0
    total_edit_distance = 0
    if len(target_formulas) != len(predicted_formulas):
        print("number of formulas doesn't match")
        return

    for tf, pf in zip(target_formulas, predicted_formulas):
        tf_ = tf.strip()
        pf_ = pf.strip()
        true_token = tf_.split(' ')
        predicted_tokens = pf_.split(' ')
        l = max(len(true_token), len(predicted_tokens))
        edit_distance = distance.levenshtein(true_token, predicted_tokens)
        total_len += l
        total_edit_distance += edit_distance
    score = 1. - float(total_edit_distance) / total_len
    print('Edit Distance Accuracy: %f' % score)
    return score


def main(args):
    parameters = process_args(args)

    target_formulas_file = parameters.target_file
    predicted_formulas_file = parameters.predicted_file

    target_formulas = open(target_formulas_file).readlines()
    predicted_formulas = open(predicted_formulas_file).readlines()

    edit_distance_eval(target_formulas, predicted_formulas)


if __name__ == '__main__':
    main(sys.argv[1:])



