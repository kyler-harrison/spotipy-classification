import statistics as stat
import matplotlib.pyplot as plt
import pandas as pd


# visualize occurence of each class in data, print some simple stats
def class_stats(data_path, class_col):
    df = pd.read_csv(data_path)

    # hacky way to get a list of artists ["a", "b", "c"] and the count of which they occur in data [5, 12, 8]
    all_classes = df[class_col].to_list()
    unique_classes = list(set(all_classes))
    class_counts = [0] * len(unique_classes)
    for cls in all_classes:
        idx = unique_classes.index(cls)
        class_counts[idx] += 1

    # make bar chart (class names get a bit messy on x-axis)
    plt.bar(unique_classes, class_counts)
    plt.xticks(rotation="vertical")
    plt.show()

    # NOTE should look at histograms (for all classes and indiv classes) of different features in data

    # simple stats
    mean_classes = stat.mean(class_counts)
    stdev_classes = stat.stdev(class_counts)
    print(f"mean num classes = {mean_classes}\nst. dev. = {stdev_classes}")


def main():
    data_path = "data/tracks.csv"
    class_col = "artist"
    class_stats(data_path, class_col)


if __name__ == "__main__":
    main()
    
