import json
import traceback


def dump_and_crash(func):
    def wrapper(batch):
        try:
            return func(batch)
        except Exception as e:
            # Dumping batch to a JSON file
            with open("./crash_dump.json", "w") as file:
                json.dump(batch, file)

            # Dumping stack trace to a text file
            with open("./crash_dump.txt", "w") as file:
                file.write(traceback.format_exc())

            # Reraising the exception
            raise e

    return wrapper
