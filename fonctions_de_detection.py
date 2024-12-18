import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import gmean
import json 
import os


#------------------------------------------------------------------------------------------------------------------


def create_event_json(event_type, file_name, annotated_times_video, annotated_times_measurements,
                      detection_method, reference_movement):
    """
    Crée un fichier JSON contenant les informations sur la détection d'événements.
    
    Parameters:
    - event_type (str): Type d'événements étudiés (parmi 'clap', 'rebond_table', 'rebond_raquette')
    - file_name (str): Nom du fichier brut correspondant (sans l'extension)
    - video_link (str): Lien vers la vidéo correspondante
    - annotated_times_video (list of float): Liste des temps d'annotations dans la vidéo
    - annotated_times_measurements (list of float): Liste des temps d'annotations dans les mesures
    - detection_method (str): Méthode de détection utilisée ('methode_1', 'methode_2', 'methode_3')
    - reference_movement (list of 2 floats) : Intervalle donnant le mouvement depuis lequel on va chercher des similitudes
    - detected_times (list of tuples): Liste numérotée des temps détectés (e.g., [(1, time1), (2, time2), ...])
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision
    - first_frame_link (str): Lien vers la frame correspondant au premier événement annoté (pas encore implémenté)
    - output_file (str): Chemin vers le fichier JSON de sortie
    
    Returns:
    - None
    """

    # Construire le chemin vers le fichier CSV brut
    file_path = os.path.join('../fichiers_bruts', f'{file_name}.csv')
    
    # Vérifier si une détection automatique doit être effectuée
    if event_type == 'rebond_raquette':

        if detection_method == 'methode_1':
            # Appeler la fonction methode_i pour calculer detected_times et precision_metrics
            detected_times, precision_metrics = methode_1(
                file_path=file_path,
                t_lines_original=annotated_times_measurements
            )
        
        if detection_method == 'methode_2':
            # Appeler la fonction methode_i pour calculer detected_times et precision_metrics
            detected_times, precision_metrics = methode_2(
                file_path=file_path,
                t_lines_original=annotated_times_measurements
            )
        
        if detection_method == 'methode_3':
            # Appeler la fonction methode_i pour calculer detected_times et precision_metrics
            detected_times, precision_metrics = methode_3(
                file_path=file_path,
                t_lines_original=annotated_times_measurements
            )

    elif event_type == 'rebond_table':
        detected_times, precision_metrics = detec_rebond_table(
                file_path=file_path,
                t_lines_original=annotated_times_measurements
            )
    
    elif event_type == 'clap':
        detected_times, precision_metrics = detec_claps(
                file_path=file_path,
                t_lines_original=annotated_times_measurements
            )



    else:
        # Si la détection automatique n'est pas applicable, lever une exception ou gérer autrement
        raise ValueError("La détection automatique n'est pas implémentée pour ces paramètres")
    
    # Convertir les temps détectés en un format sérialisable (liste de dictionnaires)
    detected_times_serializable = [{'index': idx, 'time': time} for idx, time in detected_times]
    
    # Construire le dictionnaire des données
    data = {
        'event_type': event_type,
        'file_name': file_name,
        'file_link': os.path.join('../fichiers_bruts', f'{file_name}.csv'),
        'video_link': os.path.join('../fichiers_bruts', f'{file_name}.mp4'),
        'annotated_times_video': annotated_times_video,
        'annotated_times_measurements': annotated_times_measurements,
        'detection_method': detection_method,
        'reference_movement': reference_movement,
        'detected_times': detected_times_serializable,
        'precision_metrics': precision_metrics,
        #'first_frame_link': first_frame_link
    }
    
    # Chemin vers le fichier JSON de sortie
    output_file = f'../json_files/{file_name}.json'
    
    # Écrire le dictionnaire dans un fichier JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=None)
    
    print(f"Le fichier JSON a été créé avec succès : {output_file}")








#------------------------------------------------------------------------------------------------------------------


def calculer_precision(detected_times, t_lines_original, max_allowed_time_diff=0.35):
    """
    Calcule les métriques de précision en comparant les temps détectés aux temps annotés.

    Parameters:
    - detected_times (list of float): Liste des temps détectés.
    - t_lines_original (list of float): Liste des temps annotés.
    - max_allowed_time_diff (float): Différence de temps maximale autorisée pour considérer une correspondance.

    Returns:
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision:
        - 'false_positives': Nombre de faux positifs.
        - 'true_positives': Nombre de vrais positifs.
        - 'false_negatives': Nombre de faux négatifs.
        - 'd2': Moyenne géométrique des distances.
    """
    false_negatives = 0
    differences = []
    detected_times_unmatched = detected_times.copy()

    for t_annotated in t_lines_original:
        # Trouver le temps détecté le plus proche
        best_match = None
        min_diff = max_allowed_time_diff + 1
        for t_detected in detected_times_unmatched:
            diff = abs(t_detected - t_annotated)
            if diff < min_diff:
                min_diff = diff
                best_match = t_detected
        if min_diff <= max_allowed_time_diff:
            # Correspondance trouvée
            differences.append(min_diff)
            detected_times_unmatched.remove(best_match)
        else:
            # Pas de correspondance, faux négatif
            false_negatives += 1

    false_positives = len(detected_times_unmatched)
    true_positives = len(differences)

    if differences:
        d2 = gmean(differences)
    else:
        d2 = None

    # Regrouper les métriques de précision
    precision_metrics = {
        'false_positives': false_positives,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'd2': d2
    }

    return precision_metrics

#------------------------------------------------------------------------------------------------------------------





def methode_1(file_path, t_lines_original, lowcut=15, highcut=20, order=4, quat_threshold=0.0010, t_start=0, t_end=None):
    """
    Détecte les rebonds sur la raquette de tennis de table à partir des quaternions filtrés.

    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - t_lines_original (list of float): Liste des temps de rebonds annotés.
    - lowcut (float): Fréquence de coupure basse du filtre passe-bande (en Hz).
    - highcut (float): Fréquence de coupure haute du filtre passe-bande (en Hz).
    - order (int): Ordre du filtre Butterworth.
    - quat_threshold (float): Seuil pour détecter les rebonds à partir des quaternions filtrés.
    - t_start (float): Temps de début pour l'analyse des données (en secondes).
    - t_end (float): Temps de fin pour l'analyse des données (en secondes).

    Returns:
    - detected_rebound_times (list of tuples): Liste numérotée des temps de rebonds détectés.
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision.
    """
    # Charger les données en sautant les lignes qui contiennent les métadonnées
    data = pd.read_csv(file_path, skiprows=10)

    # Ajuster le temps pour commencer à 0 secondes et convertir en secondes
    data['SampleTimeFine'] = (data['SampleTimeFine'] - data['SampleTimeFine'].iloc[0]) / 1e6

    # Filtrer les données entre t_start et t_end si t_start n'est pas zéro
    if t_start != 0 or t_end is not None:
        if t_end is None:
            t_end = data['SampleTimeFine'].iloc[-1]
        data = data[(data['SampleTimeFine'] >= t_start) & (data['SampleTimeFine'] <= t_end)]
        # Réajuster le temps pour que le début du graphique soit à 0 secondes
        data['SampleTimeFine'] = data['SampleTimeFine'] - t_start

    # Calculer la fréquence d'échantillonnage
    dt = data['SampleTimeFine'].diff().median()
    fs = 1.0 / dt

    # Vérifier que lowcut et highcut sont valides
    nyquist = 0.5 * fs
    if lowcut >= highcut:
        raise ValueError("lowcut doit être inférieur à highcut.")
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(f"Les fréquences de coupure doivent respecter 0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz).")

    # Définir le filtre passe-bande
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Appliquer le filtre passe-bande aux données d'accélération
    data['FreeAcc_X_filtered'] = filtfilt(b, a, data['FreeAcc_X'])
    data['FreeAcc_Y_filtered'] = filtfilt(b, a, data['FreeAcc_Y'])
    data['FreeAcc_Z_filtered'] = filtfilt(b, a, data['FreeAcc_Z'])

    # Appliquer le filtre passe-bande aux quaternions
    if all(col in data.columns for col in ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']):
        data['Quat_W_filtered'] = filtfilt(b, a, data['Quat_W'])
        data['Quat_X_filtered'] = filtfilt(b, a, data['Quat_X'])
        data['Quat_Y_filtered'] = filtfilt(b, a, data['Quat_Y'])
        data['Quat_Z_filtered'] = filtfilt(b, a, data['Quat_Z'])
    else:
        raise ValueError("Les colonnes des quaternions ne sont pas présentes dans les données.")

    # Détection des rebonds à partir des quaternions filtrés
    threshold = quat_threshold
    n_samples_0_05s = int(np.ceil(0.05 / dt))

    # Calculer la valeur maximale absolue parmi les quaternions filtrés à chaque instant
    max_abs_quat = data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max(axis=1)
    condition = max_abs_quat > threshold

    # Initialiser la liste des intervalles de rebonds
    intervals = []
    i = 0
    len_condition = len(condition)

    while i < len_condition:
        if condition.iloc[i]:
            # Début de l'intervalle
            interval_start = i
            j = i + 1
            while j < len_condition:
                if not condition.iloc[j]:
                    # Vérifier si la condition reste fausse pendant au moins 0.05s
                    if j + n_samples_0_05s <= len_condition:
                        if not condition.iloc[j:j + n_samples_0_05s].any():
                            # Fin de l'intervalle
                            interval_end = j
                            break
                        else:
                            j += 1
                    else:
                        # Fin des données
                        interval_end = len_condition - 1
                        break
                else:
                    j += 1
            else:
                # Fin des données sans trouver la fin de l'intervalle
                interval_end = len_condition - 1

            # Vérifier que pour tous les quaternions, le maximum de sa valeur absolue pendant l'intervalle
            # dépasse 1/10 du plus grand maximum des quaternions sur cet intervalle
            interval_data = data.iloc[interval_start:interval_end]
            max_abs_values = interval_data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max()
            max_overall = max_abs_values.max()
            if all(max_abs_values >= (max_overall / 10)):
                # Ajouter l'intervalle à la liste des rebonds
                intervals.append((interval_start, interval_end))

            # Continuer à partir de la fin de l'intervalle
            i = interval_end + n_samples_0_05s
        else:
            i += 1

    # Extraire les temps de début des intervalles de rebonds
    rebound_times = data['SampleTimeFine'].iloc[[start for start, end in intervals]].values

    # Supprimer les rebonds trop proches pour éviter les doublons
    min_interval = 0.1  # en secondes
    filtered_rebound_times = []
    if len(rebound_times) > 0:
        filtered_rebound_times = [rebound_times[0]]
        for t in rebound_times[1:]:
            if t - filtered_rebound_times[-1] >= min_interval:
                filtered_rebound_times.append(t)

    # Calcul de la précision
    precision_metrics = calculer_precision(filtered_rebound_times, t_lines_original, max_allowed_time_diff=0.3)

    # Créer la liste numérotée des temps de rebonds détectés
    detected_rebound_times = list(enumerate(filtered_rebound_times, start=1))

    return detected_rebound_times, precision_metrics








#------------------------------------------------------------------------------------------------------------------

def methode_2(file_path, t_lines_original, lowcut=15, highcut=20, order=4, quat_threshold=0.0010, higher_threshold=0.0030, t_start=0, t_end=None):
    """
    Détecte les rebonds sur la raquette de tennis de table à partir des quaternions filtrés en utilisant la méthode 2.
    
    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - t_lines_original (list of float): Liste des temps de rebonds annotés.
    - lowcut (float): Fréquence de coupure basse du filtre passe-bande (en Hz).
    - highcut (float): Fréquence de coupure haute du filtre passe-bande (en Hz).
    - order (int): Ordre du filtre Butterworth.
    - quat_threshold (float): Seuil pour détecter les rebonds à partir des quaternions filtrés.
    - higher_threshold (float): Second seuil plus élevé pour affiner la détection.
    - t_start (float): Temps de début pour l'analyse des données (en secondes).
    - t_end (float): Temps de fin pour l'analyse des données (en secondes).
    
    Returns:
    - detected_rebound_times (list of tuples): Liste numérotée des temps de rebonds détectés.
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision.
    """
    # Charger les données en sautant les lignes qui contiennent les métadonnées
    data = pd.read_csv(file_path, skiprows=10)
    
    # Ajuster le temps pour commencer à 0 secondes et convertir en secondes
    data['SampleTimeFine'] = (data['SampleTimeFine'] - data['SampleTimeFine'].iloc[0]) / 1e6
    
    # Filtrer les données entre t_start et t_end si nécessaire
    if t_start != 0 or t_end is not None:
        if t_end is None:
            t_end = data['SampleTimeFine'].iloc[-1]
        data = data[(data['SampleTimeFine'] >= t_start) & (data['SampleTimeFine'] <= t_end)]
        # Réajuster le temps pour que le début du graphique soit à 0 secondes
        data['SampleTimeFine'] = data['SampleTimeFine'] - t_start
    
    # Calculer la fréquence d'échantillonnage
    dt = data['SampleTimeFine'].diff().median()
    fs = 1.0 / dt
    
    # Vérifier que lowcut et highcut sont valides
    nyquist = 0.5 * fs
    if lowcut >= highcut:
        raise ValueError("lowcut doit être inférieur à highcut.")
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(f"Les fréquences de coupure doivent respecter 0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz).")
    
    # Définir le filtre passe-bande
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Appliquer le filtre passe-bande aux données d'accélération
    data['FreeAcc_X_filtered'] = filtfilt(b, a, data['FreeAcc_X'])
    data['FreeAcc_Y_filtered'] = filtfilt(b, a, data['FreeAcc_Y'])
    data['FreeAcc_Z_filtered'] = filtfilt(b, a, data['FreeAcc_Z'])
    
    # Appliquer le filtre passe-bande aux quaternions
    if all(col in data.columns for col in ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']):
        data['Quat_W_filtered'] = filtfilt(b, a, data['Quat_W'])
        data['Quat_X_filtered'] = filtfilt(b, a, data['Quat_X'])
        data['Quat_Y_filtered'] = filtfilt(b, a, data['Quat_Y'])
        data['Quat_Z_filtered'] = filtfilt(b, a, data['Quat_Z'])
    else:
        raise ValueError("Les colonnes des quaternions ne sont pas présentes dans les données.")
    
    # Détection des rebonds à partir des quaternions filtrés
    threshold = quat_threshold
    dt = data['SampleTimeFine'].diff().median()
    n_samples_0_05s = int(np.ceil(0.05 / dt))
    
    # Calculer la valeur maximale absolue parmi les quaternions filtrés à chaque instant
    max_abs_quat = data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max(axis=1)
    data['max_abs_quat'] = max_abs_quat
    condition = max_abs_quat > threshold
    
    # Initialiser la liste des intervalles de rebonds
    intervals = []
    adjusted_rebound_times = []
    i = 0
    len_condition = len(condition)
    
    while i < len_condition:
        if condition.iloc[i]:
            # Début de l'intervalle
            interval_start = i
            j = i + 1
            while j < len_condition:
                if not condition.iloc[j]:
                    # Vérifier si la condition reste fausse pendant au moins 0.05s
                    if j + n_samples_0_05s <= len_condition:
                        if not condition.iloc[j:j + n_samples_0_05s].any():
                            # Fin de l'intervalle
                            interval_end = j
                            break
                        else:
                            j += 1
                    else:
                        # Fin des données
                        interval_end = len_condition - 1
                        break
                else:
                    j += 1
            else:
                # Fin des données sans trouver la fin de l'intervalle
                interval_end = len_condition - 1
    
            # Traiter l'intervalle [interval_start, interval_end]
            interval_data = data.iloc[interval_start:interval_end]
            max_abs_values = interval_data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max()
            max_overall = max_abs_values.max()
            if all(max_abs_values >= (max_overall / 10)):
                intervals.append((interval_start, interval_end))
    
                # Déterminer le temps du rebond pour cet intervalle
                time_interval_start = data['SampleTimeFine'].iloc[interval_start]
    
                if interval_data['max_abs_quat'].max() > higher_threshold:
                    # Rechercher les passages par zéro de l'accélération en X
                    window_size = int(np.ceil(0.1 / dt))  # Fenêtre de 0.1 seconde
                    window_start = max(0, interval_start - window_size)
                    window_end = min(len(data), interval_end + window_size)
    
                    acc_x_window = data['FreeAcc_X'].iloc[window_start:window_end].values
                    time_window = data['SampleTimeFine'].iloc[window_start:window_end].values
    
                    # Trouver les passages par zéro
                    zero_crossings_indices = np.where(np.diff(np.sign(acc_x_window)))[0]
                    if len(zero_crossings_indices) > 0:
                        zero_crossings_indices += window_start  # Ajuster les indices
                        times_zero_crossings = data['SampleTimeFine'].iloc[zero_crossings_indices].values
    
                        # Calculer les différences de temps par rapport au début de l'intervalle
                        delta_t = times_zero_crossings - time_interval_start - 0.1
    
                        # Séparer les passages par zéro avant et après
                        mask_before = delta_t < 0
                        mask_after = delta_t >= 0
    
                        delta_t_before = -delta_t[mask_before]  # distances positives
                        times_before = times_zero_crossings[mask_before]
    
                        delta_t_after = delta_t[mask_after]
                        times_after = times_zero_crossings[mask_after]
    
                        # Trouver les passages par zéro les plus proches avant et après
                        min_delta_t_before = delta_t_before.min() if len(delta_t_before) > 0 else None
                        time_before = times_before[delta_t_before.argmin()] if len(delta_t_before) > 0 else None
    
                        min_delta_t_after = delta_t_after.min() if len(delta_t_after) > 0 else None
                        time_after = times_after[delta_t_after.argmin()] if len(delta_t_after) > 0 else None
    
                        # Appliquer la règle de décision
                        if min_delta_t_before is not None and min_delta_t_after is not None:
                            if min_delta_t_before < (1/3) * min_delta_t_after:
                                # Choisir le passage par zéro avant
                                adjusted_time = time_before
                            else:
                                # Choisir le passage par zéro après
                                adjusted_time = time_after
                        elif min_delta_t_before is not None:
                            adjusted_time = time_before
                        elif min_delta_t_after is not None:
                            adjusted_time = time_after
                        else:
                            # Aucun passage par zéro trouvé, utiliser le temps initial
                            adjusted_time = time_interval_start
                    else:
                        # Aucun passage par zéro trouvé, utiliser le temps initial
                        adjusted_time = time_interval_start
                else:
                    # Le second seuil n'est pas dépassé, utiliser le temps initial
                    adjusted_time = time_interval_start
    
                # Ajouter le temps ajusté à la liste des rebonds
                adjusted_rebound_times.append(adjusted_time)
    
            # Continuer à partir de la fin de l'intervalle
            i = interval_end + n_samples_0_05s
        else:
            i += 1
    
    # Supprimer les rebonds trop proches pour éviter les doublons
    min_interval = 0.1  # en secondes
    adjusted_rebound_times.sort()
    filtered_rebound_times = []
    for t in adjusted_rebound_times:
        if not filtered_rebound_times or t - filtered_rebound_times[-1] >= min_interval:
            filtered_rebound_times.append(t)
    
    # Calcul de la précision
    precision_metrics = calculer_precision(filtered_rebound_times, t_lines_original, max_allowed_time_diff=0.35)
    
    # Créer la liste numérotée des temps de rebonds détectés
    detected_rebound_times = list(enumerate(filtered_rebound_times, start=1))
    
    return detected_rebound_times, precision_metrics






#------------------------------------------------------------------------------------------------------------------



def methode_3(file_path, t_lines_original, lowcut=15, highcut=20, order=4, quat_threshold=0.0010):
    """
    Détecte les rebonds sur la raquette de tennis de table à partir des quaternions filtrés.

    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - t_lines_original (list of float): Liste des temps de rebonds annotés.
    - lowcut (float): Fréquence de coupure basse du filtre passe-bande (en Hz).
    - highcut (float): Fréquence de coupure haute du filtre passe-bande (en Hz).
    - order (int): Ordre du filtre Butterworth.
    - quat_threshold (float): Seuil pour détecter les rebonds à partir des quaternions filtrés.

    Returns:
    - detected_rebound_times (list of tuples): Liste numérotée des temps de rebonds détectés.
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision:
        - 'false_positives': Nombre de faux positifs.
        - 'true_positives': Nombre de vrais positifs.
        - 'false_negatives': Nombre de faux négatifs.
        - 'd2': Moyenne géométrique des distances.
    """
    import pandas as pd
    import numpy as np
    from scipy.signal import butter, filtfilt

    # Charger les données en sautant les lignes qui contiennent les métadonnées
    data = pd.read_csv(file_path, skiprows=10)

    # Ajuster le temps pour commencer à 0 secondes et convertir en secondes
    data['SampleTimeFine'] = (data['SampleTimeFine'] - data['SampleTimeFine'].iloc[0]) / 1e6

    # Calculer la fréquence d'échantillonnage
    dt = data['SampleTimeFine'].diff().median()
    fs = 1.0 / dt

    # Vérifier que lowcut et highcut sont valides
    nyquist = 0.5 * fs
    if lowcut >= highcut:
        raise ValueError("lowcut doit être inférieur à highcut.")
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(f"Les fréquences de coupure doivent respecter 0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz).")

    # Définir le filtre passe-bande
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Appliquer le filtre passe-bande aux données d'accélération
    data['FreeAcc_X_filtered'] = filtfilt(b, a, data['FreeAcc_X'])
    data['FreeAcc_Y_filtered'] = filtfilt(b, a, data['FreeAcc_Y'])
    data['FreeAcc_Z_filtered'] = filtfilt(b, a, data['FreeAcc_Z'])

    # Appliquer le filtre passe-bande aux quaternions
    if all(col in data.columns for col in ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']):
        data['Quat_W_filtered'] = filtfilt(b, a, data['Quat_W'])
        data['Quat_X_filtered'] = filtfilt(b, a, data['Quat_X'])
        data['Quat_Y_filtered'] = filtfilt(b, a, data['Quat_Y'])
        data['Quat_Z_filtered'] = filtfilt(b, a, data['Quat_Z'])
    else:
        raise ValueError("Les colonnes des quaternions ne sont pas présentes dans les données.")

    # Détection des rebonds à partir des quaternions filtrés
    threshold = quat_threshold
    n_samples_0_05s = int(np.ceil(0.05 / dt))

    # Calculer la valeur maximale absolue parmi les quaternions filtrés à chaque instant
    max_abs_quat = data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max(axis=1)
    data['max_abs_quat'] = max_abs_quat
    condition = max_abs_quat > threshold

    # Initialiser la liste des intervalles de rebonds
    intervals = []
    adjusted_rebound_times = []
    i = 0
    len_condition = len(condition)

    while i < len_condition:
        if condition.iloc[i]:
            # Début de l'intervalle
            interval_start = i
            j = i + 1
            while j < len_condition:
                if not condition.iloc[j]:
                    # Vérifier si la condition reste fausse pendant au moins 0.05s
                    if j + n_samples_0_05s <= len_condition:
                        if not condition.iloc[j:j + n_samples_0_05s].any():
                            # Fin de l'intervalle
                            interval_end = j
                            break
                        else:
                            j += 1
                    else:
                        # Fin des données
                        interval_end = len_condition - 1
                        break
                else:
                    j += 1
            else:
                # Fin des données sans trouver la fin de l'intervalle
                interval_end = len_condition - 1

            # Traiter l'intervalle [interval_start, interval_end]
            interval_data = data.iloc[interval_start:interval_end]
            max_abs_values = interval_data[['Quat_W_filtered', 'Quat_X_filtered', 'Quat_Y_filtered', 'Quat_Z_filtered']].abs().max()
            max_overall = max_abs_values.max()
            if all(max_abs_values >= (max_overall / 10)):
                intervals.append((interval_start, interval_end))

                # Déterminer le temps du rebond pour cet intervalle
                time_interval_start = data['SampleTimeFine'].iloc[interval_start]

                # Rechercher les passages par zéro où l'accélération en X passe de négatif à positif
                window_size = int(np.ceil(0.1 / dt))  # Fenêtre de 0.1 seconde
                window_start = max(0, interval_start - window_size)
                window_end = min(len(data), interval_end + window_size)

                acc_x_window = data['FreeAcc_X'].iloc[window_start:window_end].values

                # Trouver les passages par zéro de négatif à positif pour Acc_X
                positive_crossings_indices_x = np.where((acc_x_window[:-1] < 0) & (acc_x_window[1:] >= 0))[0]
                if len(positive_crossings_indices_x) > 0:
                    positive_crossings_indices_x += window_start  # Ajuster les indices
                    times_positive_crossings_x = data['SampleTimeFine'].iloc[positive_crossings_indices_x].values

                    # Trouver le passage par zéro le plus proche de time_interval_start
                    delta_t_x = times_positive_crossings_x - time_interval_start
                    adjusted_time_x = times_positive_crossings_x[np.abs(delta_t_x).argmin()]
                else:
                    # Aucun passage par zéro trouvé pour Acc_X, utiliser le temps initial
                    adjusted_time_x = time_interval_start

                # Calculer l'écart entre la détection initiale et ajustée pour Acc_X
                time_difference_x = abs(adjusted_time_x - time_interval_start)

                if time_difference_x > 0.3:
                    # Si l'écart est supérieur à 0.3s, chercher le passage par zéro pour Acc_Z
                    acc_z_window = data['FreeAcc_Z'].iloc[window_start:window_end].values

                    # Trouver les passages par zéro de négatif à positif pour Acc_Z
                    positive_crossings_indices_z = np.where((acc_z_window[:-1] < 0) & (acc_z_window[1:] >= 0))[0]
                    if len(positive_crossings_indices_z) > 0:
                        positive_crossings_indices_z += window_start  # Ajuster les indices
                        times_positive_crossings_z = data['SampleTimeFine'].iloc[positive_crossings_indices_z].values

                        # Trouver le passage par zéro le plus proche de time_interval_start
                        delta_t_z = times_positive_crossings_z - time_interval_start
                        adjusted_time_z = times_positive_crossings_z[np.abs(delta_t_z).argmin()]

                        # Utiliser adjusted_time_z comme temps ajusté
                        adjusted_time = adjusted_time_z
                    else:
                        # Aucun passage par zéro trouvé pour Acc_Z, garder adjusted_time_x
                        adjusted_time = adjusted_time_x
                else:
                    # Si l'écart est inférieur ou égal à 0.3s, utiliser adjusted_time_x
                    adjusted_time = adjusted_time_x

                # Ajouter le temps ajusté à la liste des rebonds
                adjusted_rebound_times.append(adjusted_time)

            # Continuer à partir de la fin de l'intervalle
            i = interval_end + n_samples_0_05s
        else:
            i += 1

    # Supprimer les rebonds trop proches pour éviter les doublons
    min_interval = 0.1  # en secondes
    adjusted_rebound_times.sort()
    filtered_rebound_times = []
    for t in adjusted_rebound_times:
        if not filtered_rebound_times or t - filtered_rebound_times[-1] >= min_interval:
            filtered_rebound_times.append(t)

    # Calcul de la précision
    max_allowed_time_diff = 0.35  # en secondes
    precision_metrics = calculer_precision(filtered_rebound_times, t_lines_original, max_allowed_time_diff=max_allowed_time_diff)

    # Créer la liste numérotée des temps de rebonds détectés
    detected_rebound_times = list(enumerate(filtered_rebound_times, start=1))

    return detected_rebound_times, precision_metrics



#-----------------------------------------------------------------------------------------------------------------------







def methode_3_simplified(times, freeacc_x, freeacc_y, freeacc_z, quat_w, quat_x, quat_y, quat_z, 
                         lowcut=15, highcut=20, order=4, quat_threshold=0.0010):
    """
    Détecte les rebonds sur la raquette de tennis de table à partir des quaternions filtrés.
    Cette version n'utilise pas de fichier CSV ni de données annotées.
    Elle se base uniquement sur les données passées en argument et ne calcule plus de métriques de précision.

    Parameters:
    - times (array): tableau des temps (en secondes).
    - freeacc_x, freeacc_y, freeacc_z (array): Accélérations libres en x, y, z.
    - quat_w, quat_x, quat_y, quat_z (array): Quaternions mesurés.
    - lowcut (float): Fréquence basse du filtre passe-bande.
    - highcut (float): Fréquence haute du filtre passe-bande.
    - order (int): Ordre du filtre Butterworth.
    - quat_threshold (float): Seuil pour détecter les rebonds à partir des quaternions filtrés.

    Returns:
    - detected_rebound_times (list of float): Liste des temps de rebonds détectés.
    """
    # Vérifications de base
    if len(times) == 0:
        return []

    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    nyquist = 0.5 * fs

    nyq = fs / 2.0
    if highcut >= nyq:
        highcut = nyq - 0.1
    if lowcut >= highcut:
        lowcut = highcut / 2.0
    
    if lowcut >= highcut:
        raise ValueError("lowcut doit être inférieur à highcut.")
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(f"0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz).")

    # Définir le filtre passe-bande
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Filtrer les accélérations
    freeacc_x_filt = filtfilt(b, a, freeacc_x)
    freeacc_y_filt = filtfilt(b, a, freeacc_y)
    freeacc_z_filt = filtfilt(b, a, freeacc_z)

    # Vérifier que les quaternions sont présents
    if quat_w is None or quat_x is None or quat_y is None or quat_z is None:
        raise ValueError("Les données quaternion doivent être fournies.")

    # Filtrer les quaternions
    quat_w_filt = filtfilt(b, a, quat_w)
    quat_x_filt = filtfilt(b, a, quat_x)
    quat_y_filt = filtfilt(b, a, quat_y)
    quat_z_filt = filtfilt(b, a, quat_z)

    # Calcul de la valeur max absolue parmi les quaternions filtrés à chaque instant
    max_abs_quat = np.max(np.abs(np.vstack([quat_w_filt, quat_x_filt, quat_y_filt, quat_z_filt]).T), axis=1)
    condition = max_abs_quat > quat_threshold

    n_samples_0_05s = int(np.ceil(0.05 / dt))

    detected_rebounds = []
    i = 0
    len_condition = len(condition)

    while i < len_condition:
        if condition[i]:
            # Début intervalle
            interval_start = i
            j = i + 1
            while j < len_condition:
                if not condition[j]:
                    # Vérifier la condition fausse pendant 0.05s
                    if j + n_samples_0_05s <= len_condition:
                        if not condition[j:j + n_samples_0_05s].any():
                            interval_end = j
                            break
                        else:
                            j += 1
                    else:
                        interval_end = len_condition - 1
                        break
                else:
                    j += 1
            else:
                interval_end = len_condition - 1

            # Intervalle détecté, on va trouver un temps représentatif du rebond
            # Utilisation du passage par zéro d'Acc_X comme dans la méthode originale
            interval_data_indices = range(interval_start, interval_end)
            time_interval_start = times[interval_start]

            window_size = int(np.ceil(0.1 / dt))
            window_start = max(0, interval_start - window_size)
            window_end = min(len(times), interval_end + window_size)

            acc_x_window = freeacc_x[window_start:window_end]

            positive_crossings_indices_x = np.where((acc_x_window[:-1] < 0) & (acc_x_window[1:] >= 0))[0]
            if len(positive_crossings_indices_x) > 0:
                positive_crossings_indices_x += window_start
                times_positive_crossings_x = times[positive_crossings_indices_x]

                # Trouver le passage par zéro le plus proche de time_interval_start
                delta_t_x = times_positive_crossings_x - time_interval_start
                adjusted_time_x = times_positive_crossings_x[np.abs(delta_t_x).argmin()]
            else:
                adjusted_time_x = time_interval_start

            time_difference_x = abs(adjusted_time_x - time_interval_start)

            if time_difference_x > 0.3:
                # Chercher passage par zéro sur Z
                acc_z_window = freeacc_z[window_start:window_end]
                positive_crossings_indices_z = np.where((acc_z_window[:-1] < 0) & (acc_z_window[1:] >= 0))[0]
                if len(positive_crossings_indices_z) > 0:
                    positive_crossings_indices_z += window_start
                    times_positive_crossings_z = times[positive_crossings_indices_z]

                    delta_t_z = times_positive_crossings_z - time_interval_start
                    adjusted_time_z = times_positive_crossings_z[np.abs(delta_t_z).argmin()]
                    adjusted_time = adjusted_time_z
                else:
                    adjusted_time = adjusted_time_x
            else:
                adjusted_time = adjusted_time_x

            detected_rebounds.append(adjusted_time)
            i = interval_end + n_samples_0_05s
        else:
            i += 1

    # Supprimer les rebonds trop proches
    min_interval = 0.1
    filtered_rebounds = []
    for t in sorted(detected_rebounds):
        if not filtered_rebounds or t - filtered_rebounds[-1] >= min_interval:
            filtered_rebounds.append(t)

    return filtered_rebounds


#----------------------------------------------------------------------------------------------------------------------





def detec_rebond_table(file_path, t_lines_original, window_size=5, delta_t=0.05, threshold=0.5, min_interval=0.5):
    """
    Détecte les rebonds de la balle sur la table en utilisant les données d'accélération.

    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - t_lines_original (list of float): Liste des temps de rebonds annotés.
    - window_size (int): Taille de la fenêtre pour le filtre moyenne mobile.
    - delta_t (float): Delta t pour le calcul de la dérivée (en secondes).
    - threshold (float): Seuil pour détecter les rebonds basé sur la magnitude de la dérivée (en g/s).
    - min_interval (float): Intervalle minimum entre les rebonds (en secondes).

    Returns:
    - detected_bounce_times (list of tuples): Liste numérotée des temps de rebonds détectés.
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision.
    """
    # Charger les données en sautant les lignes de métadonnées
    data = pd.read_csv(file_path, skiprows=10)

    # Ajuster le temps pour commencer à 0 secondes et convertir en secondes
    data['SampleTimeFine'] = (data['SampleTimeFine'] - data['SampleTimeFine'].iloc[0]) / 1e6

    # Appliquer un moyennage mobile aux données d'accélération
    data['FilteredAcc_X'] = data['FreeAcc_X'].rolling(window=window_size, center=False).mean()
    data['FilteredAcc_Y'] = data['FreeAcc_Y'].rolling(window=window_size, center=False).mean()
    data['FilteredAcc_Z'] = data['FreeAcc_Z'].rolling(window=window_size, center=False).mean()

    # Remplacer les valeurs NaN résultant du moyennage mobile par les valeurs originales
    data['FilteredAcc_X'].fillna(data['FreeAcc_X'], inplace=True)
    data['FilteredAcc_Y'].fillna(data['FreeAcc_Y'], inplace=True)
    data['FilteredAcc_Z'].fillna(data['FreeAcc_Z'], inplace=True)

    # Calculer la dérivée de l'accélération filtrée
    data['dAcc_X'] = data['FilteredAcc_X'].diff() / delta_t
    data['dAcc_Y'] = data['FilteredAcc_Y'].diff() / delta_t
    data['dAcc_Z'] = data['FilteredAcc_Z'].diff() / delta_t

    # Remplacer les valeurs NaN résultant de la différenciation par zéro
    data['dAcc_X'].fillna(0, inplace=True)
    data['dAcc_Y'].fillna(0, inplace=True)
    data['dAcc_Z'].fillna(0, inplace=True)

    # Calculer la magnitude de la dérivée de l'accélération
    data['dAcc_Magnitude'] = np.sqrt(data['dAcc_X']**2 + data['dAcc_Y']**2 + data['dAcc_Z']**2)

    # Détection des rebonds
    last_detection_time = -np.inf  # Initialiser le dernier temps de détection
    detected_bounce_times = []  # Liste pour stocker les temps de rebonds détectés

    # Parcourir les données pour détecter les rebonds
    for idx in data.index:
        current_time = data['SampleTimeFine'].iloc[idx]
        if data['dAcc_Magnitude'].iloc[idx] > threshold:
            if current_time - last_detection_time >= min_interval:
                detected_bounce_times.append(current_time)
                last_detection_time = current_time

    # Calculer les métriques de précision
    precision_metrics = calculer_precision(detected_bounce_times, t_lines_original, max_allowed_time_diff=0.35)

    # Créer la liste numérotée des temps de rebonds détectés
    detected_bounce_times_numbered = list(enumerate(detected_bounce_times, start=1))

    return detected_bounce_times_numbered, precision_metrics


#-----------------------------------------------------------------------------------------------------------------------





def detec_claps(file_path, t_lines_original, derivative_threshold=100, immobility_threshold=5):
    """
    Détecte les claps à partir des dérivées des accélérations.

    Parameters:
    - file_path (str): Chemin vers le fichier CSV.
    - t_lines_original (list of float): Liste des temps de claps annotés.
    - derivative_threshold (float): Seuil pour détecter les claps à partir des dérivées des accélérations.
    - immobility_threshold (float): Seuil pour considérer l'accélération comme "proche de zéro" (en m/s²).

    Returns:
    - detected_clap_times (list of tuples): Liste numérotée des temps de claps détectés.
    - precision_metrics (dict): Dictionnaire contenant les métriques de précision:
        - 'false_positives': Nombre de faux positifs.
        - 'true_positives': Nombre de vrais positifs.
        - 'false_negatives': Nombre de faux négatifs.
        - 'd2': Moyenne géométrique des distances.
    """
    # Charger les données en sautant les lignes qui contiennent les métadonnées
    data = pd.read_csv(file_path, skiprows=10)

    # Ajuster le temps pour commencer à 0 secondes et convertir en secondes
    data['SampleTimeFine'] = (data['SampleTimeFine'] - data['SampleTimeFine'].iloc[0]) / 1e6

    # Calculer la fréquence d'échantillonnage
    dt = data['SampleTimeFine'].diff().median()
    fs = 1.0 / dt

    # Utiliser delta_t pour calculer les dérivées
    delta_t = 0.1  # secondes

    # Calculer les dérivées des accélérations
    if len(data) >= 2:
        data['dAcc_X'] = (data['FreeAcc_X'] - data['FreeAcc_X'].shift(1)) / delta_t
        data['dAcc_Y'] = (data['FreeAcc_Y'] - data['FreeAcc_Y'].shift(1)) / delta_t
        data['dAcc_Z'] = (data['FreeAcc_Z'] - data['FreeAcc_Z'].shift(1)) / delta_t
    else:
        data['dAcc_X'] = 0
        data['dAcc_Y'] = 0
        data['dAcc_Z'] = 0

    # Remplacer les valeurs NaN par 0
    data['dAcc_X'].fillna(0, inplace=True)
    data['dAcc_Y'].fillna(0, inplace=True)
    data['dAcc_Z'].fillna(0, inplace=True)

    # Paramètres pour la détection des claps
    min_interval_between_claps = 0.3  # secondes
    required_time_in_immobility = 0.1  # secondes

    # Convertir les données en numpy arrays
    times = data['SampleTimeFine'].values
    acc_x = data['FreeAcc_X'].values
    acc_y = data['FreeAcc_Y'].values
    acc_z = data['FreeAcc_Z'].values
    dacc_x = data['dAcc_X'].values
    dacc_y = data['dAcc_Y'].values
    dacc_z = data['dAcc_Z'].values

    # Liste pour stocker les temps des claps détectés
    clap_times = []
    last_clap_time = -np.inf
    candidate_claps = []

    # Parcourir chaque échantillon
    for i in range(len(times)):
        t_current = times[i]
        max_abs_accel_derivative = max(abs(dacc_x[i]), abs(dacc_y[i]), abs(dacc_z[i]))

        # Détection d'un candidat clap
        if max_abs_accel_derivative > derivative_threshold:
            t_candidate = t_current

            # Condition avant le clap (immobilité de -0.13s à -0.03s)
            t_start_pre = t_candidate - 0.13
            t_end_pre = t_candidate - 0.03

            # Trouver les indices correspondants
            idx_start_pre = np.searchsorted(times, t_start_pre, side='left')
            idx_end_pre = np.searchsorted(times, t_end_pre, side='right')

            if idx_end_pre > idx_start_pre:
                acc_within_threshold = (
                    (abs(acc_x[idx_start_pre:idx_end_pre]) <= immobility_threshold) &
                    (abs(acc_y[idx_start_pre:idx_end_pre]) <= immobility_threshold) &
                    (abs(acc_z[idx_start_pre:idx_end_pre]) <= immobility_threshold)
                )

                time_in_immobility = np.sum(acc_within_threshold) * dt
                if time_in_immobility >= required_time_in_immobility:
                    # Condition avant le clap satisfaite
                    candidate_clap = {
                        't_candidate': t_candidate,
                        'status': 'pending',
                        't_start_post': t_candidate + 0.12,
                        't_end_post': t_candidate + 0.22
                    }
                    candidate_claps.append(candidate_clap)
            # Sinon, condition avant le clap non satisfaite

        # Mise à jour des claps candidats
        pending_claps = candidate_claps.copy()

        for candidate_clap in pending_claps:
            if candidate_clap['status'] == 'pending':
                t_candidate = candidate_clap['t_candidate']
                t_end_post = candidate_clap['t_end_post']

                if t_current >= t_end_post:
                    # Temps d'évaluer la condition après le clap
                    t_start_post = candidate_clap['t_start_post']

                    idx_start_post = np.searchsorted(times, t_start_post, side='left')
                    idx_end_post = np.searchsorted(times, t_end_post, side='right')

                    if idx_end_post > idx_start_post:
                        acc_within_threshold_post = (
                            (abs(acc_x[idx_start_post:idx_end_post]) <= immobility_threshold) &
                            (abs(acc_y[idx_start_post:idx_end_post]) <= immobility_threshold) &
                            (abs(acc_z[idx_start_post:idx_end_post]) <= immobility_threshold)
                        )

                        time_in_immobility_post = np.sum(acc_within_threshold_post) * dt

                        if time_in_immobility_post >= required_time_in_immobility:
                            if t_candidate - last_clap_time >= min_interval_between_claps:
                                # Clap confirmé
                                candidate_clap['status'] = 'confirmed'
                                clap_times.append(t_candidate)
                                last_clap_time = t_candidate
                            # Sinon, trop proche du clap précédent
                        # Sinon, condition après le clap non satisfaite
                    # Sinon, données insuffisantes pour la condition après le clap

                    # Retirer le clap candidat de la liste
                    candidate_claps.remove(candidate_clap)
                # Sinon, en attente de plus de données

    # Calcul de la précision
    precision_metrics = calculer_precision(clap_times, t_lines_original, max_allowed_time_diff=0.35)

    # Créer la liste numérotée des temps de claps détectés
    detected_clap_times = list(enumerate(clap_times, start=1))

    return detected_clap_times, precision_metrics











'''
file_path = '../fichiers_bruts/prot_services_marteaux.csv'
t_lines_original = [5.63, 7.82, 9.92]  # Liste des temps de rebonds annotés
detected_rebounds, metrics = methode_3(file_path, t_lines_original)

# Afficher les rebonds détectés
for idx, time in detected_rebounds:
    print(f"Rebond détecté {idx}: {time:.4f} s")

# Afficher les métriques de précision
print("Métriques de précision:")
print(f"Faux positifs: {metrics['false_positives']}")
print(f"Vrais positifs: {metrics['true_positives']}")
print(f"Faux négatifs: {metrics['false_negatives']}")
if metrics['d2'] is not None:
    print(f"d2 (moyenne géométrique des distances): {metrics['d2']:.4f} s")
else:
    print("Aucun vrai positif trouvé, impossible de calculer d2")
    '''