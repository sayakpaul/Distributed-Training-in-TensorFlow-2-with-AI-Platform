import argparse
import model_training

def get_args():
	args_parser = argparse.ArgumentParser()
	args_parser.add_argument(
		"--bucket",
		help="Name of the GCS Bucket.",
		nargs="+",
		required=True)
	args_parser.add_argument(
		"--train-pattern",
		help="Pattern of GCS paths to training data.",
		nargs="+",
		required=True)
	args_parser.add_argument(
		"--valid-pattern",
		help="Pattern of GCS paths to validation data.",
		nargs="+",
		required=True)
	return vars(args_parser.parse_args())


def main():
	args = get_args()
	model_training.run(args)


if __name__ == "__main__":
    main()