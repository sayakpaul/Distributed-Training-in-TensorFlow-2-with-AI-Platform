import argparse
import model_training

def get_args():
	args_parser = argparse.ArgumentParser()
	args_parser.add_argument(
		"--bucket",
		help="Name of the GCS Bucket.",
		required=True)
	args_parser.add_argument(
		"--train-pattern",
		help="Pattern of GCS paths to training data.",
		required=True)
	args_parser.add_argument(
		"--valid-pattern",
		help="Pattern of GCS paths to validation data.",
		required=True)
	return args_parser.parse_args()


def main():
	args = get_args()
	model_training.run(args)


if __name__ == "__main__":
    main()