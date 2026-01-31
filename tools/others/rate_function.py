import numpy as np


THRESHOLDS = {
    'task_1': {
        10: 0.999813,
        50: 0.992637,
        100: 0.962943,
        200: 0.898612,
        500: 0.709303
    },
    'task_2': {
        10: 0.99947,
        50: 0.993401,
        100: 0.974118,
        200: 0.884668,
        500: 0.640783
    },
    'task_3': {
        10: 0.942062,
        50: 0.613176,
        100: 0.413698,
        200: 0.266759,
        500: 0.147857
    },
    'task_4': {
        10: 0.99799,
        50: 0.959811,
        100: 0.779649,
        200: 0.516812,
        500: 0.251291
    }
}


def rate_probs(prob_array):
    """Probability rating function (vectorized version)"""
    prob_array = np.asarray(prob_array)
    original_shape = prob_array.shape

    if original_shape[-1] != 4:
        raise ValueError(f"The last dimension of the input array must be 4, but got {original_shape[-1]}")

    # Create rating array
    ratings = np.full(original_shape, 6, dtype=np.int32)  # Default rating is 6

    # Threshold order: 10, 50, 100, 200, 500
    threshold_values = [10, 50, 100, 200, 500]

    # Judge from strict to loose for each threshold
    for task_idx in range(4):
        task = f'task_{task_idx + 1}'

        # Get all thresholds for this task
        thresholds = np.array([THRESHOLDS[task][thr] for thr in threshold_values])

        # Compare all probabilities for this task
        task_probs = prob_array[..., task_idx]

        # Start judgment from the strictest threshold
        for i, (thr, rating) in enumerate(zip(thresholds, [1, 2, 3, 4, 5])):
            # Find probabilities greater than current threshold and set to corresponding rating
            mask = task_probs > thr

            # For positions that have already been rated (previous stricter threshold satisfied), do not modify
            # So start from positions with rating 6
            if i == 0:
                ratings[..., task_idx][mask] = rating
            else:
                # Only modify positions where rating is still 6 (no previous threshold satisfied) and current condition is met
                ratings[..., task_idx][np.logical_and(mask, ratings[..., task_idx] == 6)] = rating

    return ratings