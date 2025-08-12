import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets.utils import connect_to_database, fetch_metadata, load_pbs_scores


class BCIC2b(torch.utils.data.Dataset):
    num_subjects = 9

    def __init__(self, args, target_subject, is_test=False, transform=None):

        self.is_subject_independent = args.is_subject_independent
        self.mode = args.mode

        db_path = os.path.join(args.DB_PATH, "bcic2b.db")
        con = connect_to_database(db_path)
        cur = con.cursor()

        query = self._build_query(target_subject, is_test)
        self.metadata = fetch_metadata(cur, query)
        print("LOG >>> Successfully connected to the database")

        self.transform = transform

        # Load PBS scores if applicable
        self.pbs_score = (
            load_pbs_scores(args.model, "bcic2b") if args.litmodel == "pbs" else None
        )

        cur.close()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.metadata[idx][-1]
        raw = np.load(filename)

        data = raw["data"][np.newaxis, ...]
        label = np.array(raw["label"], dtype=np.int64)
        label = F.one_hot(torch.tensor(label), num_classes=4).float()

        if self.transform:
            data, label = self.transform(data, label)

        if self.pbs_score is not None:
            return data, label, self.pbs_score[self.metadata[idx][0] - 1]

        return data, label

    def _build_query(self, target_subject, is_test):
        if self.mode == "cls":
            if self.is_subject_independent:
                if is_test:
                    return (
                        f"SELECT * FROM MetaData WHERE MetaData.Sub == {target_subject}"
                    )
                else:
                    train_subject_list = [
                        i for i in range(self.num_subjects) if i != target_subject
                    ]
                    conditions = " OR ".join(
                        f"MetaData.Sub == {sub}" for sub in train_subject_list
                    )
                    return f"SELECT * FROM MetaData WHERE {conditions}"
            else:
                test_flag = 1 if is_test else 0
                return f"SELECT * FROM MetaData WHERE MetaData.Sub == {target_subject} AND MetaData.Test == {test_flag}"
        elif self.mode == "ssl":
            return "SELECT * FROM MetaData"
        else:
            raise ValueError("Mode should be either 'cls' or 'ssl'")
