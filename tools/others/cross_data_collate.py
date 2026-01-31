def processed_unlabeled_submodel(train_label_mask, test_label_mask, label_matrix, smiles_list):
    import numpy as np
    import copy
    from collections import defaultdict
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import random

    # === Fix random seed for reproducibility ===
    np.random.seed(42)
    random.seed(42)

    k = 5
    smiles_list = np.array(smiles_list)
    fold_train_mask = {f: copy.deepcopy(np.zeros(train_label_mask.shape)) for f in range(k)}
    fold_valid_mask = {f: copy.deepcopy(np.zeros(train_label_mask.shape)) for f in range(k)}
    fold_train_submodel_mask = {
        fold: {sub: copy.deepcopy(np.zeros(train_label_mask.shape)) for sub in range(k)}
        for fold in range(k)
    }

    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    # === Globally record used molecule indices in unlabeled data ===
    global_used_unlabeled_indices = set()

    # === Multi-task loop ===
    for task_idx in range(23, 30):
        print(f"\n=== Processing task {task_idx} ===")

        other_tasks = list(range(7, 23))

        # ---- Get unlabeled / positive samples ----
        unlabeled_indices = np.where(
            (train_label_mask[:, other_tasks].sum(axis=1) > 0)
            & (train_label_mask[:, task_idx] == 0)
        )[0]
        positive_indices = np.where(train_label_mask[:, task_idx] == 1)[0]

        print(f"Task {task_idx}: original unlabeled={len(unlabeled_indices)}, positive={len(positive_indices)}")

        # ---- Build scaffold groups ----
        def group_by_scaffold(indices):
            groups = defaultdict(list)
            for idx in indices:
                smiles = smiles_list[idx]
                scaffold = get_scaffold(smiles)
                groups[scaffold].append(idx)
            return groups

        unlabeled_groups = group_by_scaffold(unlabeled_indices)
        positive_groups = group_by_scaffold(positive_indices)

        # ---- Step 1: Test set (unlabeled) ----
        target_test_size = 1000
        test_indices = []
        all_scaffolds = list(unlabeled_groups.keys())
        np.random.shuffle(all_scaffolds)

        for scaffold in all_scaffolds:
            if len(test_indices) >= target_test_size:
                break
            group = unlabeled_groups[scaffold]
            # Filter out globally used samples
            available = [i for i in group if i not in global_used_unlabeled_indices]
            if not available:
                continue
            chosen_idx = random.choice(available)
            test_indices.append(chosen_idx)
            global_used_unlabeled_indices.add(chosen_idx)

        for idx in test_indices:
            test_label_mask[idx, task_idx] = 1

        print(f"Task {task_idx} - Test set: {len(test_indices)} samples")

        # ---- Step 2: Validation set (unlabeled) ----
        target_valid_size = 555
        fold_valid_indices = {fold: [] for fold in range(k)}

        remaining_unlabeled = [
            i for i in unlabeled_indices if i not in global_used_unlabeled_indices
        ]
        np.random.shuffle(remaining_unlabeled)

        fold_idx = 0
        for idx in remaining_unlabeled:
            if all(len(v) >= target_valid_size for v in fold_valid_indices.values()):
                break
            while len(fold_valid_indices[fold_idx]) >= target_valid_size:
                fold_idx = (fold_idx + 1) % k
            fold_valid_indices[fold_idx].append(idx)
            global_used_unlabeled_indices.add(idx)
            fold_idx = (fold_idx + 1) % k

        for fold in range(k):
            print(f"Task {task_idx} - Fold {fold} validation set: {len(fold_valid_indices[fold])}")

        # ---- Step 3: Positive samples (complete scaffold split) ----
        positive_scaffolds = list(positive_groups.keys())
        np.random.shuffle(positive_scaffolds)
        fold_positive_valid = {fold: [] for fold in range(k)}
        fold_positive_train = {fold: [] for fold in range(k)}

        for i, scaffold in enumerate(positive_scaffolds):
            fold = i % k
            # All molecules in scaffold enter this fold's validation set
            fold_positive_valid[fold].extend(positive_groups[scaffold])
            # Other folds use these scaffold molecules as training set
            for other_fold in range(k):
                if other_fold != fold:
                    fold_positive_train[other_fold].extend(positive_groups[scaffold])

        # ---- Step 4: Training set (unlabeled + labeled) ----
        remaining_train_unlabeled = [
            i for i in unlabeled_indices if i not in global_used_unlabeled_indices
        ]
        np.random.shuffle(remaining_train_unlabeled)

        for fold in range(k):
            pos_train = fold_positive_train[fold]
            target_unlabeled_size = len(pos_train) * 5

            fold_submodel_indices = {sub: [] for sub in range(k)}
            for i, idx in enumerate(remaining_train_unlabeled):
                sub = i % k
                if len(fold_submodel_indices[sub]) < target_unlabeled_size:
                    fold_submodel_indices[sub].append(idx)

            # Update mask for each submodel
            for sub in range(k):
                sub_indices = pos_train + fold_submodel_indices[sub]
                for idx in sub_indices:
                    fold_train_submodel_mask[fold][sub][idx, task_idx] = 1
                print(f"Task {task_idx} - Fold {fold}, Submodel {sub}: labeled={len(pos_train)}, unlabeled={len(fold_submodel_indices[sub])}")

            # Main training set mask (positive + unlabeled)
            all_train_indices = list(set(pos_train + remaining_train_unlabeled))
            for idx in all_train_indices:
                fold_train_mask[fold][idx, task_idx] = 1

            # Validation set mask (positive + unlabeled)
            valid_indices = fold_positive_valid[fold] + fold_valid_indices[fold]
            for idx in valid_indices:
                fold_valid_mask[fold][idx, task_idx] = 1

            print(f"Task {task_idx} - Fold {fold}: training={len(all_train_indices)}, validation={len(valid_indices)}")

        print(f"Task {task_idx} - Split completed, remaining unlabeled samples: {len([i for i in unlabeled_indices if i not in global_used_unlabeled_indices])}")

    print("\n=== All tasks processed (no within-task overlap version) ===")
    return fold_train_mask, fold_valid_mask, fold_train_submodel_mask, test_label_mask