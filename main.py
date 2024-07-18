import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from classifiers.gaussian_classifier import GaussianClassifier
from classifiers.classifier_GLIA import classifier_GLIA
from classifiers.global_IA import global_IA_clf
from data.data_loader import extract_bands_glacier, load_data
from data.data_preprocessor import prepare_training_data, prep_X_IA
from data.data_statistics import calculate_slopes_and_intercepts
from utils.metrics import calculate_accuracy
from utils.visualization import compare_confusion_matrices, plot_classification_results
from sklearn.metrics import confusion_matrix, accuracy_score
import os

abbreviation_dict = {
    'firn': 'Firn',
    'superimposedice': 'SI',
    'glacierice_textured1': 'GI1',
    'glacierice_textured2': 'GI2',
    'crevasse' : 'crevasse'
}

def main(training_dir, glacier_scene_path, zone_titles, cm, std):
    
    ### 1. Getting slopes to train classifiers + confusion matrices
    df, bands_by_zone, zone_bands = load_data(training_dir, zone_titles, std)

    #Cleaning zone_bands
    for zone in zone_titles:
        zone_df = df[df['Zone'] == zone]
        zone_df = zone_df.sort_values(by='Date')
        zone_df = zone_df.drop_duplicates(subset=['Date'])
        zone_df = zone_df[zone_df['IA_mean'] > 5]
        zone_bands[zone] = zone_df

    X, y = prepare_training_data(bands_by_zone, zone_titles)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)


    ## ---- Classifiers
    # Gaussian
    classifier_gaussian = GaussianClassifier()
    classifier_gaussian.fit(X_train, y_train)
    y_pred, _ = classifier_gaussian.predict(X_test)
    accuracy = round(calculate_accuracy(y_test, y_pred), 2)
    print(f"Gaussian Classifier Accuracy: {accuracy}")

   

    ## IA Correction
    average_year_data, slopes_dict = calculate_slopes_and_intercepts(zone_bands, zone_titles, std)
    print("slopes are", slopes_dict)
    X_IA,y = prep_X_IA(bands_by_zone, zone_titles, slopes_dict)
    X_IA_train, X_IA_test, y_train_IA, y_test_IA = train_test_split(X_IA, y, test_size=0.5)

    X_train_IA  = X_IA_train[:,0:-1]
    IA_train = X_IA_train[:,-1]
    X_test_IA   = X_IA_test[:,0:-1]
    IA_test  = X_IA_test[:,-1]

     ## Global IA
    classifier_global_IA = global_IA_clf()
    classifier_global_IA.fit(X_train_IA, y_train_IA, IA_train)
    y_pred_global_IA, p_global_IA = classifier_global_IA.predict(X_test_IA, IA_test)
    accuracy_global = round(calculate_accuracy(y_test_IA, y_pred_global_IA), 2)
    print(f"Accuracy: {accuracy_global}")

    ##GLIA
    classifier_gaussian_IA = classifier_GLIA()
    classifier_gaussian_IA.fit(X_train_IA, y_train_IA, IA_train)
    y_pred_IA, _ = classifier_gaussian_IA.predict(X_test_IA, IA_test)
    accuracy_IA = round(accuracy_score(y_test_IA, y_pred_IA),2)
    print(f"GLIA Accuracy: {accuracy_IA}")

    #Visualise with confusion matrices
    target_names = [abbreviation_dict.get(zone, zone) for zone in zone_titles]
    accuracies = [accuracy, accuracy_global, accuracy_IA]
    save_direct = "C:\\Users\MarionD\Desktop\AGP\Thesis\Classification_project\Results"
    if cm:
        compare_confusion_matrices(y_test, y_pred, y_test_IA, y_pred_global_IA, y_pred_IA, target_names, accuracies, save_dir=save_direct, std=std)


    ### 2. On glacier
    pixels, pixels_IA, valid_pixels, original_shape = extract_bands_glacier(glacier_scene_path, std)
    
    y_pred_scene, _ = classifier_gaussian.predict(pixels)
    labels_image = np.full(original_shape, -1, dtype=int)
    labels_image[valid_pixels] = y_pred_scene

    y_pred_IA_global, p_IA_global = classifier_global_IA.predict(pixels, pixels_IA)
    labels_image_global = np.full(original_shape, -1, dtype=int)
    labels_image_global[valid_pixels] = y_pred_IA_global

    y_pred_IA, p_IA = classifier_gaussian_IA.predict(pixels, pixels_IA)
    labels_image_IA = np.full(original_shape, -1, dtype=int)
    labels_image_IA[valid_pixels] = y_pred_IA
    

    titles = ["Gaussian classifier", "Global IA correction", "GLIA correction"]
    save_glaciers = "C:\\Users\MarionD\Desktop\AGP\Thesis\Classification_project\Results\Glaciers"
    base_path = 'E:\\THESIS_PRODUCTS\IW_1peryear'
    scene_path = os.path.relpath(glacier_scene_path, start=base_path)
    
    save_direct_glaciers = os.path.join(save_glaciers,scene_path)
    os.makedirs(save_direct_glaciers, exist_ok=True)
    plot_classification_results(labels_image, labels_image_global, labels_image_IA, titles, target_names, save_direct_glaciers, scene_path, std)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and classify glacier scenes.")
    parser.add_argument("glacier_scene_path", type=str, help="Path to the glacier scene to classify")
    parser.add_argument("zone_titles", nargs='+', help="List of zone titles")
    parser.add_argument("cm", nargs='+', help="Plot confusion matrices of training data (False by default)")
    parser.add_argument("std", nargs='+', help="Include std as feature, T/F")

    training_dir = 'E:\\THESIS_PRODUCTS\\subsets'
    args = parser.parse_args()
    main(training_dir, args.glacier_scene_path, args.zone_titles, args.cm, args.std)
