def cross_fold_validation(classifier, frame, folds):
    
    features = frame.drop(columns='label').copy()
    labels = frame['label'].copy()
    total = []
    scores = cross_val_score(classifier, features, labels, cv=folds, scoring='accuracy')
    for i in scores:
        total.append(i)
    return total

unique_data = unique_data.fillna(0)

def one_hot(frame, columns):
    final_frame = frame.copy()
    for column in columns:
        labels = []
        for label in final_frame[column]:
            if label in labels:
                continue
            else:
                labels.append(label)
            dict1 = {f'{column}: {x}': [] for x in labels}
            for row in final_frame[column]:
                for x in labels:
                    dict1[f'{column}: {x}'].append(1 if x in row else 0)
            one_hot_frame = pandas.DataFrame.from_dict(dict1, dtype=int)
            final_frame = pandas.concat([final_frame.reset_index(drop=True),
            one_hot_frame.reset_index(drop=True)], axis=1)
            final_frame[one_hot_frame.columns] = final_frame[one_hot_frame.columns].astype(int)
            final_frame = final_frame.drop(columns=column)

        return final_frame

cols = ['col_02', 'col_03']
unique_data = one_hot(unique_data, cols)


my_classifiers_scores = []
for classifier in my_classifiers:
    accuracy_scores = cross_fold_validation(classifier, unique_data, 5)
    my_classifiers_scores.append(accuracy_scores)
    print("Classifier: %s, Accuracy: %s." % (type(classifier).__name__, accuracy_scores))
