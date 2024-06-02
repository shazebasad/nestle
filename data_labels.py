# label data dictionaries
education_labels = {
    'Education': [1, 2, 3, 4, 5, 6, 7, 8],
    'Education_Label': [
        'No formal education',
        'Primary school',
        'Secondary school',
        'High school diploma or equivalent (e.g., GED)',
        'Some college or vocational training',
        'Bachelor\'s degree',
        'Master\'s degree',
        'Doctorate or professional degree'
    ]
}

income_labels = {
    'Income': [1, 2, 3, 4, 5, 6, 7],
    'Income_Label': [
        'up to 5k',
        '5k - 10k',
        '10k-20k',
        '20k-40k',
        '40k-60k',
        '60k-80k',
        '80k+'
    ]
}

gender_labels = {
    'Gender': [1, 2],
    'Gender_Label': ['Male', 'Female']
}

family_status_labels = {
    'Family status': [1, 2, 3, 4, 5, 6],
    'Family_Status_Label': [
        'Single',
        'Married',
        'Divorced',
        'Widowed',
        'Separated',
        'Domestic Partnership/Civil Union'
    ]
}

age_labels = {
    'Age': [1, 2, 3, 4, 5, 6],
    'Age_Label': [
        '18-25',
        '25-35',
        '35-45',
        '45-55',
        '55-65',
        '65+'
    ]
}

# Combined label data dictionary
label_data_dict = {
    'Education': education_labels,
    'Income': income_labels,
    'Gender': gender_labels,
    'Family status': family_status_labels,
    'Age': age_labels
}
