import json
import os

import matplotlib.pyplot as plt
import numpy as np


def processei(file_paths, tolerance, max_iterations):
    list1, list2, list3 = [], [], []

    # # Read data from provided file paths
    # for file_path in file_paths:
    #     df = pd.read_excel(file_path)
    #     list1.extend(df.get('IEWXH', []).tolist())
    #     list2.extend(df.get('IEWJM', []).tolist())
    #     list3.extend(df.get('IETSY', []).tolist())
    # 读取 JSONL 文件中的 E/I 维度标签
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                annotation = data.get("annotation", {}).get("E/I", {})
                list1.append(annotation.get("A1", ""))  # 获取 A1 的 E/I 标签
                list2.append(annotation.get("A2", ""))  # 获取 A2 的 E/I 标签
                list3.append(annotation.get("A3", ""))  # 获取 A3 的 E/I 标签

    # Mapping categories to numerical values
    category_mapping = {"E+": 0, "E-": 1, "I-": 2, "I+": 3}
    z_init = []

    # Determine the middle value based on the sorted mapped values
    for i in range(len(list1)):
        mapped_values = [category_mapping[list1[i]], category_mapping[list2[i]], category_mapping[list3[i]]]
        middle_mapped_value = sorted(mapped_values)[1]
        original_category = [key for key, value in category_mapping.items() if value == middle_mapped_value][0]
        z_init.append(original_category)

    # Initialize count matrix
    matrix_size = len(category_mapping)
    count_matrix = np.zeros((3, matrix_size, matrix_size), dtype=int)

    # Populate count matrix with category data
    for i in range(len(list1)):
        cat1 = category_mapping[list1[i]]
        cat2 = category_mapping[list2[i]]
        cat3 = category_mapping[list3[i]]
        z_cat = category_mapping[z_init[i]]
        count_matrix[0, z_cat, cat1] += 1
        count_matrix[1, z_cat, cat2] += 1
        count_matrix[2, z_cat, cat3] += 1

    # Add smoothing by incrementing all values by 1
    count_matrix += 1

    # Calculate counts and probabilities
    count0 = np.sum(count_matrix[:, 0, :])
    count1 = np.sum(count_matrix[:, 1, :])
    count2 = np.sum(count_matrix[:, 2, :])
    count3 = np.sum(count_matrix[:, 3, :])
    count = count0 + count1 + count2 + count3
    PE1, PE2 = count0 / count, count1 / count
    PI1, PI2 = count2 / count, count3 / count
    PE = PE1 + PE2
    PI = PI1 + PI2

    # Function to get probability values
    def get_pr_values(value, matrix, index):
        if value == "E+":
            return matrix[index][0][0], matrix[index][1][0]
        elif value == "E-":
            return matrix[index][0][1], matrix[index][1][1]
        elif value == "I-":
            return matrix[index][0][2], matrix[index][1][2]
        elif value == "I+":
            return matrix[index][0][3], matrix[index][1][3]

    # Initialize the result based on categories
    initial_resultes = [1 if z in ["E+", "E-"] else 0 for z in z_init]

    # Create a new matrix to update probabilities
    new_matrix = np.zeros((3, 2, 4))
    for i in range(3):
        new_matrix[i, 0] = ((count_matrix[i, 0] * PE1) + (count_matrix[i, 1] * PE2)) / PE
        new_matrix[i, 1] = ((count_matrix[i, 2] * PI1) + (count_matrix[i, 3] * PI2)) / PI

    iteration = 0
    previous_resultes = initial_resultes

    # Function to calculate result probabilities
    def calculate_resulte(list1, list2, list3, matrix, PE, PI):
        resultes = []
        for i in range(len(list1)):
            pr1e, pr1i = get_pr_values(list1[i], matrix, 0)
            pr2e, pr2i = get_pr_values(list2[i], matrix, 1)
            pr3e, pr3i = get_pr_values(list3[i], matrix, 2)
            product_e = pr1e * pr2e * pr3e * PE
            product_i = pr1i * pr2i * pr3i * PI
            total = product_e + product_i
            resulte = product_e / (total if total != 0 else 1)
            resultes.append(resulte)
        return resultes

    # Iteratively update results until convergence or reaching max iterations
    while iteration < max_iterations:
        resultes = calculate_resulte(list1, list2, list3, new_matrix, PE, PI)
        PE = sum([1 for r in resultes if r > 0.5]) / len(resultes)
        PI = 1 - PE
        diff = np.abs(np.array(resultes) - np.array(previous_resultes)).max()
        if diff < tolerance:
            break
        previous_resultes = resultes

        # Update count matrix with new results
        new_count_matrix = np.zeros((3, 2, 4), dtype=float)
        for i in range(len(list1)):
            cat1 = category_mapping[list1[i]]
            cat2 = category_mapping[list2[i]]
            cat3 = category_mapping[list3[i]]
            new_count_matrix[0, 0, cat1] += resultes[i]
            new_count_matrix[0, 1, cat1] += 1 - resultes[i]
            new_count_matrix[1, 0, cat2] += resultes[i]
            new_count_matrix[1, 1, cat2] += 1 - resultes[i]
            new_count_matrix[2, 0, cat3] += resultes[i]
            new_count_matrix[2, 1, cat3] += 1 - resultes[i]

        # Normalize the new matrix
        new_matrix = np.copy(new_count_matrix)
        for layer in range(new_matrix.shape[0]):
            for row in range(new_matrix.shape[1]):
                row_sum = np.sum(new_matrix[layer, row])
                if row_sum > 0:
                    new_matrix[layer, row] = new_matrix[layer, row] / row_sum

        iteration += 1

    # Final results calculation
    final_results = []
    for i in range(len(list1)):
        pr1e, pr1i = get_pr_values(list1[i], new_matrix, 0)
        pr2e, pr2i = get_pr_values(list2[i], new_matrix, 1)
        pr3e, pr3i = get_pr_values(list3[i], new_matrix, 2)
        product_e = pr1e * pr2e * pr3e * PE
        product_i = pr1i * pr2i * pr3i * PI
        total = product_e + product_i
        resulte = product_e / total
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        final_results.append((combination, resulte))

    # Deduplicate and sort final results
    final_results = list(set(final_results))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # Frequency analysis of combinations
    frequency_dict = {}
    for i in range(len(list1)):
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        frequency_dict[combination] = frequency_dict.get(combination, 0) + 1

    final_results_with_freq = [
        (combination, resulte, frequency_dict.get(combination, 0)) for combination, resulte in final_results
    ]

    # Sort results and find the median
    final_results_with_freq.sort(key=lambda x: x[1])
    mid_index = next(
        (i for i, item in enumerate(final_results_with_freq) if item[1] >= 0.5), len(final_results_with_freq)
    )
    mid_combination = ("MID", 0.5, 0)
    final_results_with_freq.insert(mid_index, mid_combination)

    # Split and calculate cumulative frequencies
    left_combinations = final_results_with_freq[:mid_index]
    right_combinations = final_results_with_freq[mid_index + 1 :]
    cumulative_frequencies_left = []
    cumulative_frequencies_right = []

    # Left cumulative frequencies
    previous_value = 0
    for freq in reversed([item[2] for item in left_combinations]):
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_left.append(averaged_value)
        previous_value = cumulative_value
    cumulative_frequencies_left = cumulative_frequencies_left[::-1]

    # Right cumulative frequencies
    previous_value = 0
    for freq in [item[2] for item in right_combinations]:
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_right.append(averaged_value)
        previous_value = cumulative_value

    # Normalize cumulative frequencies
    max_left = max(cumulative_frequencies_left) if cumulative_frequencies_left else 1
    max_right = max(cumulative_frequencies_right) if cumulative_frequencies_right else 1
    normalized_frequencies_left = [0.5 - (x / (2 * max_left)) for x in cumulative_frequencies_left]
    normalized_frequencies_right = [(x / (2 * max_right)) + 0.5 for x in cumulative_frequencies_right]

    # Combine normalized frequencies
    combined_combinations = [item[0] for item in final_results_with_freq]
    normalized_combined_frequencies = normalized_frequencies_left + [0.5] + normalized_frequencies_right

    return combined_combinations, normalized_combined_frequencies


def processsn(file_paths, tolerance, max_iterations):
    list1, list2, list3 = [], [], []

    # Read data from provided file paths
    # for file_path in file_paths:
    #     df = pd.read_excel(file_path)
    #     list1.extend(df.get('SNWXH', []).tolist())
    #     list2.extend(df.get('SNWJM', []).tolist())
    #     list3.extend(df.get('SNTSY', []).tolist())
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                annotation = data.get("annotation", {}).get("S/N", {})
                list1.append(annotation.get("A1", ""))  # 获取 A1 的 E/I 标签
                list2.append(annotation.get("A2", ""))  # 获取 A2 的 E/I 标签
                list3.append(annotation.get("A3", ""))  # 获取 A3 的 E/I 标签
    # Mapping categories to numerical values
    category_mapping = {"S+": 0, "S-": 1, "N-": 2, "N+": 3}
    z_init = []

    # Determine the middle value based on the sorted mapped values
    for i in range(len(list1)):
        mapped_values = [category_mapping[list1[i]], category_mapping[list2[i]], category_mapping[list3[i]]]
        middle_mapped_value = sorted(mapped_values)[1]
        original_category = [key for key, value in category_mapping.items() if value == middle_mapped_value][0]
        z_init.append(original_category)

    # Initialize count matrix
    matrix_size = len(category_mapping)
    count_matrix = np.zeros((3, matrix_size, matrix_size), dtype=int)

    # Populate count matrix with category data
    for i in range(len(list1)):
        cat1 = category_mapping[list1[i]]
        cat2 = category_mapping[list2[i]]
        cat3 = category_mapping[list3[i]]
        z_cat = category_mapping[z_init[i]]
        count_matrix[0, z_cat, cat1] += 1
        count_matrix[1, z_cat, cat2] += 1
        count_matrix[2, z_cat, cat3] += 1

    # Add smoothing by incrementing all values by 1
    count_matrix += 1

    # Calculate counts and probabilities
    count0 = np.sum(count_matrix[:, 0, :])
    count1 = np.sum(count_matrix[:, 1, :])
    count2 = np.sum(count_matrix[:, 2, :])
    count3 = np.sum(count_matrix[:, 3, :])
    count = count0 + count1 + count2 + count3
    PE1, PE2 = count0 / count, count1 / count
    PI1, PI2 = count2 / count, count3 / count
    PE = PE1 + PE2
    PI = PI1 + PI2

    # Function to get probability values
    def get_pr_values(value, matrix, index):
        if value == "S+":
            return matrix[index][0][0], matrix[index][1][0]
        elif value == "S-":
            return matrix[index][0][1], matrix[index][1][1]
        elif value == "N-":
            return matrix[index][0][2], matrix[index][1][2]
        elif value == "N+":
            return matrix[index][0][3], matrix[index][1][3]

    # Initialize the result based on categories
    initial_resultes = [1 if z in ["S+", "S-"] else 0 for z in z_init]

    # Create a new matrix to update probabilities
    new_matrix = np.zeros((3, 2, 4))
    for i in range(3):
        new_matrix[i, 0] = ((count_matrix[i, 0] * PE1) + (count_matrix[i, 1] * PE2)) / PE
        new_matrix[i, 1] = ((count_matrix[i, 2] * PI1) + (count_matrix[i, 3] * PI2)) / PI

    iteration = 0
    previous_resultes = initial_resultes

    # Function to calculate result probabilities
    def calculate_resulte(list1, list2, list3, matrix, PE, PI):
        resultes = []
        for i in range(len(list1)):
            pr1e, pr1i = get_pr_values(list1[i], matrix, 0)
            pr2e, pr2i = get_pr_values(list2[i], matrix, 1)
            pr3e, pr3i = get_pr_values(list3[i], matrix, 2)
            product_e = pr1e * pr2e * pr3e * PE
            product_i = pr1i * pr2i * pr3i * PI
            total = product_e + product_i
            resulte = product_e / (total if total != 0 else 1)
            resultes.append(resulte)
        return resultes

    # Iteratively update results until convergence or reaching max iterations
    while iteration < max_iterations:
        resultes = calculate_resulte(list1, list2, list3, new_matrix, PE, PI)
        PE = sum([1 for r in resultes if r > 0.5]) / len(resultes)
        PI = 1 - PE
        diff = np.abs(np.array(resultes) - np.array(previous_resultes)).max()
        if diff < tolerance:
            break
        previous_resultes = resultes

        # Update count matrix with new results
        new_count_matrix = np.zeros((3, 2, 4), dtype=float)
        for i in range(len(list1)):
            cat1 = category_mapping[list1[i]]
            cat2 = category_mapping[list2[i]]
            cat3 = category_mapping[list3[i]]
            new_count_matrix[0, 0, cat1] += resultes[i]
            new_count_matrix[0, 1, cat1] += 1 - resultes[i]
            new_count_matrix[1, 0, cat2] += resultes[i]
            new_count_matrix[1, 1, cat2] += 1 - resultes[i]
            new_count_matrix[2, 0, cat3] += resultes[i]
            new_count_matrix[2, 1, cat3] += 1 - resultes[i]

        # Normalize the new matrix
        new_matrix = np.copy(new_count_matrix)
        for layer in range(new_matrix.shape[0]):
            for row in range(new_matrix.shape[1]):
                row_sum = np.sum(new_matrix[layer, row])
                if row_sum > 0:
                    new_matrix[layer, row] = new_matrix[layer, row] / row_sum

        iteration += 1

    # Final results calculation
    final_results = []
    for i in range(len(list1)):
        pr1e, pr1i = get_pr_values(list1[i], new_matrix, 0)
        pr2e, pr2i = get_pr_values(list2[i], new_matrix, 1)
        pr3e, pr3i = get_pr_values(list3[i], new_matrix, 2)
        product_e = pr1e * pr2e * pr3e * PE
        product_i = pr1i * pr2i * pr3i * PI
        total = product_e + product_i
        resulte = product_e / total
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        final_results.append((combination, resulte))

    # Deduplicate and sort final results
    final_results = list(set(final_results))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # Frequency analysis of combinations
    frequency_dict = {}
    for i in range(len(list1)):
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        frequency_dict[combination] = frequency_dict.get(combination, 0) + 1

    final_results_with_freq = [
        (combination, resulte, frequency_dict.get(combination, 0)) for combination, resulte in final_results
    ]

    # Sort results and find the median
    final_results_with_freq.sort(key=lambda x: x[1])
    mid_index = next(
        (i for i, item in enumerate(final_results_with_freq) if item[1] >= 0.5), len(final_results_with_freq)
    )
    mid_combination = ("MID", 0.5, 0)
    final_results_with_freq.insert(mid_index, mid_combination)

    # Split and calculate cumulative frequencies
    left_combinations = final_results_with_freq[:mid_index]
    right_combinations = final_results_with_freq[mid_index + 1 :]
    cumulative_frequencies_left = []
    cumulative_frequencies_right = []

    # Left cumulative frequencies
    previous_value = 0
    for freq in reversed([item[2] for item in left_combinations]):
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_left.append(averaged_value)
        previous_value = cumulative_value
    cumulative_frequencies_left = cumulative_frequencies_left[::-1]

    # Right cumulative frequencies
    previous_value = 0
    for freq in [item[2] for item in right_combinations]:
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_right.append(averaged_value)
        previous_value = cumulative_value

    # Normalize cumulative frequencies
    max_left = max(cumulative_frequencies_left) if cumulative_frequencies_left else 1
    max_right = max(cumulative_frequencies_right) if cumulative_frequencies_right else 1
    normalized_frequencies_left = [0.5 - (x / (2 * max_left)) for x in cumulative_frequencies_left]
    normalized_frequencies_right = [(x / (2 * max_right)) + 0.5 for x in cumulative_frequencies_right]

    # Combine normalized frequencies
    combined_combinations = (
        [item[0] for item in left_combinations] + [mid_combination[0]] + [item[0] for item in right_combinations]
    )
    combined_frequencies = normalized_frequencies_left + [mid_combination[2]] + normalized_frequencies_right

    # Filter out the 'MID' combination
    filtered_combinations = [comb for comb in combined_combinations if comb != "MID"]
    filtered_frequencies = [freq for comb, freq in zip(combined_combinations, combined_frequencies) if comb != "MID"]

    # Sort combined results by frequency
    sorted_results = sorted(zip(filtered_combinations, filtered_frequencies), key=lambda x: x[1])
    sorted_combinations, sorted_frequencies = zip(*sorted_results) if sorted_results else ([], [])

    # Plot the results
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_combinations, sorted_frequencies, color="skyblue")

    # Add title and labels
    plt.axhline(y=0.5, color="r", linestyle="--", label="Result = 0.5")
    plt.xlabel("Combination", fontsize=20)
    plt.ylabel("Soft Label", fontsize=20)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and show plot
    output_path = "./SN_F.png"
    # plt.savefig(output_path)
    # plt.show()

    return sorted_combinations, sorted_frequencies


def processtf(file_paths, tolerance, max_iterations):
    # Initialize lists to store data from each column
    list1, list2, list3 = [], [], []

    # Read data from the provided file paths and extend lists
    # for file_path in file_paths:
    #     df = pd.read_excel(file_path)
    #     list1.extend(df.get('TFWXH', []).tolist())
    #     list2.extend(df.get('TFWJM', []).tolist())
    #     list3.extend(df.get('TFTSY', []).tolist())
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                annotation = data.get("annotation", {}).get("T/F", {})
                list1.append(annotation.get("A1", ""))  # 获取 A1 的 E/I 标签
                list2.append(annotation.get("A2", ""))  # 获取 A2 的 E/I 标签
                list3.append(annotation.get("A3", ""))  # 获取 A3 的 E/I 标签
    # Map the categories to numerical values
    category_mapping = {"T+": 0, "T-": 1, "F-": 2, "F+": 3}
    z_init = []

    # Determine the middle value based on sorted mapped values
    for i in range(len(list1)):
        mapped_values = [category_mapping[list1[i]], category_mapping[list2[i]], category_mapping[list3[i]]]
        middle_mapped_value = sorted(mapped_values)[1]
        original_category = [key for key, value in category_mapping.items() if value == middle_mapped_value][0]
        z_init.append(original_category)

    # Initialize the count matrix for counts across categories
    matrix_size = len(category_mapping)
    count_matrix = np.zeros((3, matrix_size, matrix_size), dtype=int)

    # Populate count matrix with category data
    for i in range(len(list1)):
        cat1 = category_mapping[list1[i]]
        cat2 = category_mapping[list2[i]]
        cat3 = category_mapping[list3[i]]
        z_cat = category_mapping[z_init[i]]
        count_matrix[0, z_cat, cat1] += 1
        count_matrix[1, z_cat, cat2] += 1
        count_matrix[2, z_cat, cat3] += 1

    # Add smoothing by incrementing all values by 1
    count_matrix += 1

    # Calculate counts and probabilities for each category
    count0 = np.sum(count_matrix[:, 0, :])
    count1 = np.sum(count_matrix[:, 1, :])
    count2 = np.sum(count_matrix[:, 2, :])
    count3 = np.sum(count_matrix[:, 3, :])
    count = count0 + count1 + count2 + count3
    PE1, PE2 = count0 / count, count1 / count
    PI1, PI2 = count2 / count, count3 / count
    PE = PE1 + PE2
    PI = PI1 + PI2

    # Helper function to retrieve probability values based on category
    def get_pr_values(value, matrix, index):
        if value == "T+":
            return matrix[index][0][0], matrix[index][1][0]
        elif value == "T-":
            return matrix[index][0][1], matrix[index][1][1]
        elif value == "F-":
            return matrix[index][0][2], matrix[index][1][2]
        elif value == "F+":
            return matrix[index][0][3], matrix[index][1][3]

    # Initialize results based on category
    initial_resultes = [1 if z in ["T+", "T-"] else 0 for z in z_init]

    # Create a new matrix to update probabilities
    new_matrix = np.zeros((3, 2, 4))
    for i in range(3):
        new_matrix[i, 0] = ((count_matrix[i, 0] * PE1) + (count_matrix[i, 1] * PE2)) / PE
        new_matrix[i, 1] = ((count_matrix[i, 2] * PI1) + (count_matrix[i, 3] * PI2)) / PI

    iteration = 0
    previous_resultes = initial_resultes

    # Calculate result probabilities for each iteration
    def calculate_resulte(list1, list2, list3, matrix, PE, PI):
        resultes = []
        for i in range(len(list1)):
            pr1e, pr1i = get_pr_values(list1[i], matrix, 0)
            pr2e, pr2i = get_pr_values(list2[i], matrix, 1)
            pr3e, pr3i = get_pr_values(list3[i], matrix, 2)
            product_e = pr1e * pr2e * pr3e * PE
            product_i = pr1i * pr2i * pr3i * PI
            total = product_e + product_i
            resulte = product_e / (total if total != 0 else 1)
            resultes.append(resulte)
        return resultes

    # Iteratively update results until convergence or reaching max iterations
    while iteration < max_iterations:
        resultes = calculate_resulte(list1, list2, list3, new_matrix, PE, PI)
        PE = sum([1 for r in resultes if r > 0.5]) / len(resultes)
        PI = 1 - PE
        diff = np.abs(np.array(resultes) - np.array(previous_resultes)).max()
        if diff < tolerance:
            break
        previous_resultes = resultes

        # Update the count matrix based on new results
        new_count_matrix = np.zeros((3, 2, 4), dtype=float)
        for i in range(len(list1)):
            cat1 = category_mapping[list1[i]]
            cat2 = category_mapping[list2[i]]
            cat3 = category_mapping[list3[i]]
            new_count_matrix[0, 0, cat1] += resultes[i]
            new_count_matrix[0, 1, cat1] += 1 - resultes[i]
            new_count_matrix[1, 0, cat2] += resultes[i]
            new_count_matrix[1, 1, cat2] += 1 - resultes[i]
            new_count_matrix[2, 0, cat3] += resultes[i]
            new_count_matrix[2, 1, cat3] += 1 - resultes[i]

        # Normalize the new matrix
        new_matrix = np.copy(new_count_matrix)
        for layer in range(new_matrix.shape[0]):
            for row in range(new_matrix.shape[1]):
                row_sum = np.sum(new_matrix[layer, row])
                if row_sum > 0:
                    new_matrix[layer, row] = new_matrix[layer, row] / row_sum

        iteration += 1

    # Calculate the final results
    final_results = []
    for i in range(len(list1)):
        pr1e, pr1i = get_pr_values(list1[i], new_matrix, 0)
        pr2e, pr2i = get_pr_values(list2[i], new_matrix, 1)
        pr3e, pr3i = get_pr_values(list3[i], new_matrix, 2)
        product_e = pr1e * pr2e * pr3e * PE
        product_i = pr1i * pr2i * pr3i * PI
        total = product_e + product_i
        resulte = product_e / total
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        final_results.append((combination, resulte))

    # Remove duplicates and sort results
    final_results = list(set(final_results))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # Perform frequency analysis of combinations
    frequency_dict = {}
    for i in range(len(list1)):
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        frequency_dict[combination] = frequency_dict.get(combination, 0) + 1

    # Prepare results with frequency
    final_results_with_freq = [
        (combination, resulte, frequency_dict.get(combination, 0)) for combination, resulte in final_results
    ]

    # Sort results by soft label value
    final_results_with_freq.sort(key=lambda x: x[1])

    # Find and insert the median value (resulte = 0.5)
    mid_index = next(
        (i for i, item in enumerate(final_results_with_freq) if item[1] >= 0.5), len(final_results_with_freq)
    )
    mid_combination = ("MID", 0.5, 0)
    final_results_with_freq.insert(mid_index, mid_combination)

    # Split the results into left and right of 0.5
    left_combinations = final_results_with_freq[:mid_index]
    right_combinations = final_results_with_freq[mid_index + 1 :]

    # Calculate cumulative frequencies for both left and right sides
    cumulative_frequencies_left = []
    cumulative_frequencies_right = []

    # Calculate cumulative frequencies for the left side
    previous_value = 0
    for freq in reversed([item[2] for item in left_combinations]):
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_left.append(averaged_value)
        previous_value = cumulative_value
    cumulative_frequencies_left = cumulative_frequencies_left[::-1]

    # Calculate cumulative frequencies for the right side
    previous_value = 0
    for freq in [item[2] for item in right_combinations]:
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_right.append(averaged_value)
        previous_value = cumulative_value

    # Normalize cumulative frequencies for plotting
    max_left = max(cumulative_frequencies_left) if cumulative_frequencies_left else 1
    max_right = max(cumulative_frequencies_right) if cumulative_frequencies_right else 1
    normalized_frequencies_left = [0.5 - (x / (2 * max_left)) for x in cumulative_frequencies_left]
    normalized_frequencies_right = [(x / (2 * max_right)) + 0.5 for x in cumulative_frequencies_right]

    # Combine combinations and their frequencies for plotting
    combined_combinations = (
        [item[0] for item in left_combinations] + [mid_combination[0]] + [item[0] for item in right_combinations]
    )
    combined_frequencies = normalized_frequencies_left + [mid_combination[2]] + normalized_frequencies_right

    # Filter out the 'MID' combination for plotting
    filtered_combinations = [comb for comb in combined_combinations if comb != "MID"]
    filtered_frequencies = [freq for comb, freq in zip(combined_combinations, combined_frequencies) if comb != "MID"]

    # Sort results by frequency for consistency
    sorted_results = sorted(zip(filtered_combinations, filtered_frequencies), key=lambda x: x[1])

    # Extract sorted combinations and frequencies
    sorted_combinations, sorted_frequencies = zip(*sorted_results) if sorted_results else ([], [])

    # Plot the normalized cumulative frequencies
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_combinations, sorted_frequencies, color="skyblue")

    # Add title and labels
    plt.axhline(y=0.5, color="r", linestyle="--", label="Result = 0.5")
    plt.xlabel("Combination", fontsize=20)
    plt.ylabel("Soft Label", fontsize=20)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and show the plot
    output_path = "./TF_F.png"
    # plt.savefig(output_path)
    # plt.show()

    return sorted_combinations, sorted_frequencies


def processjp(file_paths, tolerance, max_iterations):
    # Initialize empty lists to hold the values from the Excel files
    list1, list2, list3 = [], [], []

    # Loop through each file and extract the specified columns
    # for file_path in file_paths:
    #     df = pd.read_excel(file_path)
    #     list1.extend(df.get('JPWXH', []).tolist())
    #     list2.extend(df.get('JPWJM', []).tolist())
    #     list3.extend(df.get('JPTSY', []).tolist())
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                annotation = data.get("annotation", {}).get("J/P", {})
                list1.append(annotation.get("A1", ""))  # 获取 A1 的 E/I 标签
                list2.append(annotation.get("A2", ""))  # 获取 A2 的 E/I 标签
                list3.append(annotation.get("A3", ""))  # 获取 A3 的 E/I 标签
    # Define the mapping of categories to integers
    category_mapping = {"J+": 0, "J-": 1, "P-": 2, "P+": 3}
    z_init = []

    # Map each combination of categories to its middle value
    for i in range(len(list1)):
        mapped_values = [category_mapping[list1[i]], category_mapping[list2[i]], category_mapping[list3[i]]]
        middle_mapped_value = sorted(mapped_values)[1]
        original_category = [key for key, value in category_mapping.items() if value == middle_mapped_value][0]
        z_init.append(original_category)

    # Initialize count matrix
    matrix_size = len(category_mapping)
    count_matrix = np.zeros((3, matrix_size, matrix_size), dtype=int)

    # Populate count matrix with the mapped category values
    for i in range(len(list1)):
        cat1 = category_mapping[list1[i]]
        cat2 = category_mapping[list2[i]]
        cat3 = category_mapping[list3[i]]
        z_cat = category_mapping[z_init[i]]
        count_matrix[0, z_cat, cat1] += 1
        count_matrix[1, z_cat, cat2] += 1
        count_matrix[2, z_cat, cat3] += 1

    # Add smoothing to avoid zero counts
    count_matrix += 1

    # Calculate initial proportions for each category
    count0 = np.sum(count_matrix[:, 0, :])
    count1 = np.sum(count_matrix[:, 1, :])
    count2 = np.sum(count_matrix[:, 2, :])
    count3 = np.sum(count_matrix[:, 3, :])
    count = count0 + count1 + count2 + count3
    PE1, PE2 = count0 / count, count1 / count
    PI1, PI2 = count2 / count, count3 / count
    PE = PE1 + PE2
    PI = PI1 + PI2

    # Function to retrieve PR values based on a category
    def get_pr_values(value, matrix, index):
        if value == "J+":
            return matrix[index][0][0], matrix[index][1][0]
        elif value == "J-":
            return matrix[index][0][1], matrix[index][1][1]
        elif value == "P-":
            return matrix[index][0][2], matrix[index][1][2]
        elif value == "P+":
            return matrix[index][0][3], matrix[index][1][3]

    # Initialize results based on the initial z_init values
    initial_resultes = [1 if z in ["J+", "J-"] else 0 for z in z_init]

    # Create new matrix with normalized counts for each category
    new_matrix = np.zeros((3, 2, 4))
    for i in range(3):
        new_matrix[i, 0] = ((count_matrix[i, 0] * PE1) + (count_matrix[i, 1] * PE2)) / PE
        new_matrix[i, 1] = ((count_matrix[i, 2] * PI1) + (count_matrix[i, 3] * PI2)) / PI

    iteration = 0
    previous_resultes = initial_resultes

    # Function to calculate results for each iteration
    def calculate_resulte(list1, list2, list3, matrix, PE, PI):
        resultes = []
        for i in range(len(list1)):
            pr1e, pr1i = get_pr_values(list1[i], matrix, 0)
            pr2e, pr2i = get_pr_values(list2[i], matrix, 1)
            pr3e, pr3i = get_pr_values(list3[i], matrix, 2)
            product_e = pr1e * pr2e * pr3e * PE
            product_i = pr1i * pr2i * pr3i * PI
            total = product_e + product_i
            resulte = product_e / (total if total != 0 else 1)
            resultes.append(resulte)
        return resultes

    # Iteratively update the results until convergence
    while iteration < max_iterations:
        resultes = calculate_resulte(list1, list2, list3, new_matrix, PE, PI)
        PE = sum([1 for r in resultes if r > 0.5]) / len(resultes)
        PI = 1 - PE
        diff = np.abs(np.array(resultes) - np.array(previous_resultes)).max()
        if diff < tolerance:
            break
        previous_resultes = resultes

        # Update the count matrix based on new results
        new_count_matrix = np.zeros((3, 2, 4), dtype=float)
        for i in range(len(list1)):
            cat1 = category_mapping[list1[i]]
            cat2 = category_mapping[list2[i]]
            cat3 = category_mapping[list3[i]]
            new_count_matrix[0, 0, cat1] += resultes[i]
            new_count_matrix[0, 1, cat1] += 1 - resultes[i]
            new_count_matrix[1, 0, cat2] += resultes[i]
            new_count_matrix[1, 1, cat2] += 1 - resultes[i]
            new_count_matrix[2, 0, cat3] += resultes[i]
            new_count_matrix[2, 1, cat3] += 1 - resultes[i]

        # Normalize the count matrix
        new_matrix = np.copy(new_count_matrix)
        for layer in range(new_matrix.shape[0]):
            for row in range(new_matrix.shape[1]):
                row_sum = np.sum(new_matrix[layer, row])
                if row_sum > 0:
                    new_matrix[layer, row] = new_matrix[layer, row] / row_sum

        iteration += 1

    # Generate the final results with their frequencies
    final_results = []
    for i in range(len(list1)):
        pr1e, pr1i = get_pr_values(list1[i], new_matrix, 0)
        pr2e, pr2i = get_pr_values(list2[i], new_matrix, 1)
        pr3e, pr3i = get_pr_values(list3[i], new_matrix, 2)
        product_e = pr1e * pr2e * pr3e * PE
        product_i = pr1i * pr2i * pr3i * PI
        total = product_e + product_i
        resulte = product_e / total
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        final_results.append((combination, resulte))

    # Remove duplicates and sort by result value
    final_results = list(set(final_results))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # Count the frequency of each combination
    frequency_dict = {}
    for i in range(len(list1)):
        combination = f"{list1[i]}{list2[i]}{list3[i]}"
        frequency_dict[combination] = frequency_dict.get(combination, 0) + 1

    # Attach frequencies to final results
    final_results_with_freq = [
        (combination, resulte, frequency_dict.get(combination, 0)) for combination, resulte in final_results
    ]

    # Sort results by result value and insert 'MID' at 0.5
    final_results_with_freq.sort(key=lambda x: x[1])
    mid_index = next(
        (i for i, item in enumerate(final_results_with_freq) if item[1] >= 0.5), len(final_results_with_freq)
    )
    mid_combination = ("MID", 0.5, 0)
    final_results_with_freq.insert(mid_index, mid_combination)

    # Separate left and right of 'MID'
    left_combinations = final_results_with_freq[:mid_index]
    right_combinations = final_results_with_freq[mid_index + 1 :]

    # Calculate cumulative frequencies for left and right of 'MID'
    cumulative_frequencies_left = []
    cumulative_frequencies_right = []

    previous_value = 0
    for freq in reversed([item[2] for item in left_combinations]):
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_left.append(averaged_value)
        previous_value = cumulative_value
    cumulative_frequencies_left = cumulative_frequencies_left[::-1]

    previous_value = 0
    for freq in [item[2] for item in right_combinations]:
        cumulative_value = previous_value + freq
        averaged_value = (previous_value + cumulative_value) / 2
        cumulative_frequencies_right.append(averaged_value)
        previous_value = cumulative_value

    # Normalize the cumulative frequencies
    max_left = max(cumulative_frequencies_left) if cumulative_frequencies_left else 1
    max_right = max(cumulative_frequencies_right) if cumulative_frequencies_right else 1
    normalized_frequencies_left = [0.5 - (x / (2 * max_left)) for x in cumulative_frequencies_left]
    normalized_frequencies_right = [(x / (2 * max_right)) + 0.5 for x in cumulative_frequencies_right]

    # Combine results for plotting
    combined_combinations = (
        [item[0] for item in left_combinations] + [mid_combination[0]] + [item[0] for item in right_combinations]
    )
    combined_frequencies = normalized_frequencies_left + [mid_combination[2]] + normalized_frequencies_right

    # Filter out 'MID' for plotting
    filtered_combinations = [comb for comb in combined_combinations if comb != "MID"]
    filtered_frequencies = [freq for comb, freq in zip(combined_combinations, combined_frequencies) if comb != "MID"]

    # Sort the final combinations and frequencies
    sorted_results = sorted(zip(filtered_combinations, filtered_frequencies), key=lambda x: x[1])

    # Extract sorted combinations and frequencies
    sorted_combinations, sorted_frequencies = zip(*sorted_results) if sorted_results else ([], [])

    # Plot the final bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_combinations, sorted_frequencies, color="skyblue")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Result = 0.5")
    plt.xlabel("Combination", fontsize=20)
    plt.ylabel("Soft Label", fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    output_path = "./JP_F.png"
    # plt.savefig(output_path)
    # plt.show()

    return sorted_combinations, sorted_frequencies


if __name__ == "__main__":
    file_path = "dataset/mbtibench-nolabel.jsonl"

    sorted_combinationsei, sorted_frequenciesei = processei([file_path], tolerance=1e-1, max_iterations=10000)
    sorted_combinationssn, sorted_frequenciessn = processsn([file_path], tolerance=1e-1, max_iterations=10000)
    sorted_combinationstf, sorted_frequenciestf = processtf([file_path], tolerance=1e-1, max_iterations=10000)
    sorted_combinationsjp, sorted_frequenciesjp = processjp([file_path], tolerance=1e-1, max_iterations=10000)

    results_dict = {
        "E/I": dict(zip(sorted_combinationsei, sorted_frequenciesei)),
        "S/N": dict(zip(sorted_combinationssn, sorted_frequenciessn)),
        "T/F": dict(zip(sorted_combinationstf, sorted_frequenciestf)),
        "J/P": dict(zip(sorted_combinationsjp, sorted_frequenciesjp)),
    }

    with open(file_path) as f:
        dataset_nolabel = [json.loads(line.strip()) for line in f]

    dims = ["E/I", "S/N", "T/F", "J/P"]
    dataset_with_softlabel = []
    for data in dataset_nolabel:
        softlabels, hardlabels = {}, {}
        for dim in dims:
            annotation = "".join(data["annotation"][dim].values())
            softlabel = 1 - results_dict[dim][annotation]
            hardlabel = dim[0] if annotation.count(dim[0]) >= 2 else dim[-1]
            softlabels[dim] = softlabel
            hardlabels[dim] = hardlabel
        dataset_with_softlabel.append({**data, "softlabels": softlabels, "hardlabels": hardlabels})

    if not os.path.exists("dataset/mbtibench.jsonl"):
        with open("dataset/mbtibench.jsonl", "w") as f:
            for data in dataset_with_softlabel:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
