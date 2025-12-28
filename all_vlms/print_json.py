import csv


def csv_to_models(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["LLM"].strip()
            print(f"{{'model': \"{name}\"}},")


# usage
csv_to_models("table.csv")
