import argparse
import csv
import gzip

BASEPATH = "/datasets01/ccmatrix.v1/bitexts"
THRESHOLD = 1.07
DATACAP = 10_000_000


def main(args):
    source_lines = []
    target_lines = []
    total_counter = 0
    with gzip.open(
        f"{args.basepath}/{args.src}-{args.tgt}.bitextf.tsv.gz", "rt", encoding="utf8"
    ) as f:
        tsv_reader = csv.reader((line.replace("\0", "") for line in f), delimiter="\t")

        for row in tsv_reader:
            # there are lines that seem slightly corrupted
            if "\n" in row[1]:
                new_rows = row[1].split("\n")
                for sub_row in new_rows[1:-1]:
                    sub_row = sub_row.split("\t")
                    if float(sub_row[0]) > args.threshold:
                        source_lines.append(sub_row[1])
                        target_lines.append(sub_row[2])
                    total_counter += 1
            else:
                total_counter += 1
                if float(row[0]) > args.threshold:
                    # more corrupted line problems
                    if len(row) < 3:
                        source = row[1].split("\t")[0]
                        target = row[1].split("\t")[1]
                    else:
                        source = row[1]
                        target = row[2]

                    source_lines.append(source)
                    target_lines.append(target)

    assert len(source_lines) == len(target_lines)
    print(
        f"Remaining Lines = {len(source_lines)} out of {total_counter} original lines. Percentage kept = {float(len(source_lines))/total_counter}"
    )
    print(f"Writing out to {args.outpath}")

    with open(f"{args.outpath}/ccmatrix.{args.src}-{args.tgt}.{args.src}", "w") as o:
        for line in source_lines[0 : args.datacap]:
            o.write(line.strip() + "\n")

    with open(f"{args.outpath}/ccmatrix.{args.src}-{args.tgt}.{args.tgt}", "w") as o:
        for line in target_lines[0 : args.datacap]:
            o.write(line.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src", required=True)
    parser.add_argument("-t", "--tgt", required=True)

    parser.add_argument("-o", "--outpath", required=True)

    parser.add_argument("-r", "--threshold", default=THRESHOLD)
    parser.add_argument("-b", "--basepath", default=BASEPATH)
    parser.add_argument("-d", "--datacap", default=DATACAP)

    args = parser.parse_args()
    main(args)
