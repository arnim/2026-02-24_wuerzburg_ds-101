from sklearn.model_selection import train_test_split

text_train, text_test, y_train, y_test = train_test_split(
    text, y, test_size=0.25, random_state=42, stratify=y
)