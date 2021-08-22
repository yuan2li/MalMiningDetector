import os
from feature_engineering import *
from model_construction import *
from sklearn.model_selection import train_test_split

data_path = '/mnt/diskd/datasets/mining'
base_path = './idata'
pic_path = './pic'
number = 20

if __name__ == "__main__":
    # Feature engineering
    # Data preprocessing
    black_id = os.listdir(f'{data_path}/1_2000_black')
    white_id = os.listdir(f'{data_path}/1_4000_white')
    with open(f'{base_path}/black.txt', 'w') as fp:
        fp.write('\n'.join(black_id))
    with open(f'{base_path}/white.txt', 'w') as fp:
        fp.write('\n'.join(white_id))

    get_training_data('1_2000_black', 'black')
    print('Black...done!')
    get_training_data('1_4000_white', 'white')
    print('White...done!')
    black = pd.read_csv(f'{base_path}/black.csv')
    white = pd.read_csv(f'{base_path}/white.csv')
    merge(black, white)
    print('Training data...ready!')

    # Exploratory data analysis
    plot_wordcloud(black, 'black')
    plot_wordcloud(white, 'white')
    display_top_strings(black, number, 'black')
    display_top_strings(white, number, 'white')

    # Feature vectorizing
    df = pd.read_csv(f'{base_path}/train_data.csv')
    tfidf_vectorize(df)

    # Feature visualization
    with open(f'{base_path}/tfidf_features.pkl', 'rb') as fp:
        tfidf_features = joblib.load(fp)
    with open(f'{base_path}/train_labels.pkl', 'rb') as fp:
        labels = joblib.load(fp)
    tsne_plot(tfidf_features, labels)
    print('feature engineering...done!')

    # Model construction
    # Model training
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)
    with open(f'{base_path}/train_test_set.pkl', 'wb') as fp:
        joblib.dump((X_train, X_test, y_train, y_test), fp)

    # with open(f'{base_path}/train_test_set.pkl', 'rb') as fp:
    #     X_train, X_test, y_train, y_test = joblib.load(fp)
    y_xgb_list = []
    seeds = []
    for i in np.random.randint(50, size=9):
        y_xgb = get_xgb_oof(X_train, y_train, X_test, i)
        y_xgb_list.append(y_xgb)
        seeds.append(i)
    y_xgb_array = np.array(y_xgb_list)
    with open(f'{base_path}/xgb_pred.pkl', 'wb') as fp:
        joblib.dump(y_xgb_array, fp)
    with open(f'{base_path}/seeds.pkl', 'wb') as fp:
        joblib.dump(seeds, fp)

    # with open(f'{base_path}/seeds.pkl', 'rb') as fp:
    #     seeds = joblib.load(fp)
    y_xgb_adv_list = []
    for i in seeds:
        y_xgb_adv = get_xgb_oof_advanced(X_train, y_train, X_test, i)
        y_xgb_adv_list.append(y_xgb_adv)
    y_xgb_adv_array = np.array(y_xgb_adv_list)
    with open(f'{base_path}/xgb_pred_advanced.pkl', 'wb') as fp:
        joblib.dump(y_xgb_adv_array, fp)

    # Model evaluation
    # with open(f'{base_path}/xgb_pred.pkl', 'rb') as fp:
    #     y_xgb_array = joblib.load(fp)
    # with open(f'{base_path}/xgb_pred_advanced.pkl', 'rb') as fp:
    #     y_xgb_adv_array = joblib.load(fp)
    plot_roc_curve(y_test, y_xgb_array, seeds, 3, 3, 8, 6, advanced=False)
    plot_roc_curve(y_test, y_xgb_adv_array, seeds, 3, 3, 8, 6, advanced=True)
    plot_varied_curve()
