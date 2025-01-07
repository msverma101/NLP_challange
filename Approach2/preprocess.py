import pandas as pd

from process_parallel_text import preprocess_text, merge_comments


def process_df(path):

    df = pd.read_csv(
        path,
        header=None,
        names=["CommentID", "Context", "Label", "Comment"],
    )

    df = df.dropna(subset=["Comment"])
    # Apply preprocessing to all comments
    df["Comment"] = df["Comment"].apply(preprocess_text)
    # Remove rows with empty comments
    df = df[df["Comment"].str.strip() != ""].reset_index(drop=True)


    def process_group(group):
        """
        Process a group of parallel comments and return a single processed comment.
        """

        comments = group["Comment"].tolist()

        try:
            processed_comment = merge_comments(comments)
        except Exception as e:
            print(e)
            print(group)
            print(*comments, sep="\n")
            exit()
        return processed_comment


    final_comments = (
        df.groupby("CommentID", sort=False, group_keys=False)
        .apply(process_group)
        .reset_index(name="FinalComment")
    )
    df = (
        df.drop("Comment", axis=1)
        .drop_duplicates(subset=["CommentID"])
        .merge(final_comments, on="CommentID")
    )

    return df

training_df = process_df("data/training.csv")
training_df.to_csv("data/training_preprocessed.csv", index=False)

validation_df = process_df("data/validation.csv")
validation_df.to_csv("data/validation_preprocessed.csv", index=False)
