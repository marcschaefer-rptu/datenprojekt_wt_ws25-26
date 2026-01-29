# ============================================================
# Stunden verschwendet mit diesem Skript: 64 + 42 Kaffe Pausen☕
# ============================================================
#
#    ██╗    ██╗████████╗     ██████╗ ██████╗ ██╗   ██╗██████╗ ██████╗ ███████╗     ██╗    ██╗
#    ██║    ██║╚══██╔══╝    ██╔════╝ ██╔══██╗██║   ██║██╔══██╗██╔══██╗██╔════╝    ███║    ██║
#    ██║ █╗ ██║   ██║       ██║  ███╗██████╔╝██║   ██║██████╔╝██████╔╝█████╗      ╚██║    ██║
#    ██║███╗██║   ██║       ██║   ██║██╔══██╗██║   ██║██╔═══╝ ██╔═══╝ ██╔══╝       ██║    ╚═╝
#    ╚███╔███╔╝   ██║       ╚██████╔╝██║  ██║╚██████╔╝██║     ██║     ███████╗     ██║    ██╗
#     ╚══╝╚══╝    ╚═╝        ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝     ╚══════╝     ╚═╝    ╚═╝
#                                                                                            
# ============================================================
# ASKII Art by https://patorjk.com 


import os
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import matplotlib.ticker as mpl_tick
from statistics import mean, median, mode, multimode
import numpy as np
import csv
from scipy.optimize import curve_fit

# Basisverzeichnis bestimmen (wo das Skript liegt)
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"pfad: {script_dir}")

# ============================================================
# MONATKONVERTIERUNG
# ============================================================
# Konvertiert Monatsnamen <-> Zahlen (1-12)

MONATE = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4,
    "Mai": 5, "Juni": 6, "Juli": 7, "August": 8,
    "September": 9, "Oktober": 10, "November": 11, "Dezember": 12
}

MONATE_ZAHLEN = {v: k for k, v in MONATE.items()}

def monthToInt(month):
    """Konvertiert Monatsnamen zu Zahl (1-12)
    z.B.: 'Januar' -> 1, 'Dezember' -> 12"""
    return MONATE.get(month, -1)
        
def intToMonth(month):
    """Konvertiert Zahl zu Monatsnamen (1-12)
    z.B.: 1 -> 'Januar', 12 -> 'Dezember'"""
    return MONATE_ZAHLEN.get(month, -1)

# ============================================================
# STATISTISCHE FUNKTIONEN
# ============================================================

def abweichungMedian(data):
    """Mittlere Absolute Abweichung vom Median (MAD)
    Formel: MAD = (1/n) * Σ|x_i - median|
    Misst die durchschnittliche Abweichung der Werte vom Median"""
    med = median(data)
    return sum(abs(x - med) for x in data) / len(data)

def quartile(data):
    """Berechne Quartile (25%, 50%, 75%, 100%)
    Q1 = 25. Perzentil, Q2 = Median, Q3 = 75. Perzentil, Q4 = Maximum
    Teilt Datensatz in 4 gleiche Teile"""
    sortiert = sorted(data)
    n = len(data)
    return [
        sortiert[n // 4],
        sortiert[n // 2],
        sortiert[int(n * 0.75)],
        sortiert[-1]
    ]

def dezile(data):
    """Berechne alle 9 Dezile (10%, 20%, ..., 90%)
    Nutzt numpy.percentile für konsistente Berechnung
    D_i = i*10-tes Perzentil (z.B. d5 = 50. Perzentil = Median)"""
    return [np.percentile(data, i * 10) for i in range(1, 10)]

def variationsKoeffizient(data):
    """Variationskoeffizient (CV) - Relative Streuung
    Formel: CV = σ / μ = Standardabweichung / Mittelwert
    Normalisierte Maßzahl für die Variabilität (0-1 oder in %)"""
    return np.std(data) / mean(data)

def korrelationsKoeffizient(data, data2):
    """Korrelationskoeffizient - Lineare Abhängigkeit zweier Variablen
    Formel: ρ = Cov(X,Y) / (σ_X * σ_Y)
    Wertebereich: -1 (negativ) bis +1 (positiv korreliert)"""
    return np.cov(data, data2) / (np.std(data) * np.std(data2))

# ============================================================
# DATENLADEN UND VERARBEITUNG
# ============================================================

def readFromDataToArray(data, array, dataSet):
    """Liest CSV-Daten in Python-Array (unterschiedlich für jeden Datensatz)
    - DataSet 1: Jahr, Monat, Elektrizitätserzeugung (MWh)
    - DataSet 2: Jahr, Monat, Beschäftigte (%)
    - DataSet 3: Ort, Ankünfte, Übernachtungen
    - DataSet 4: Monat, Tag, Wochentag, Anzahl Schritte"""
    i = 0
    for zeile in data:
        i += 1
        try:
            temp = zeile.strip().split(";")
            
            if dataSet == 1:
                if i > 2 and len(temp) >= 4:
                    jahr, monat, wert = int(temp[0]), monthToInt(temp[1]), int(temp[3])
                    array.append([jahr, monat, wert])
                    
            elif dataSet == 2:
                if i > 2:
                    jahr = int(temp[0]) if temp[0] else 0
                    monat = monthToInt(temp[1])
                    wert = float(temp[2]) if temp[2] else 0.0
                    wert = 0.0 if wert != wert else wert  # NaN-Check
                    array.append([jahr, monat, wert])
                    
            elif dataSet == 3:
                array.append([temp[0], int(temp[1]), monthToInt(temp[2])])
                
            elif dataSet == 31:
                if i > 2:
                    array.append([temp[0], int(temp[1]), int(temp[2])])
                    
            elif dataSet == 4:
                if i > 1 and len(temp) >= 4:
                    monat = monthToInt(temp[0])
                    tag = int(temp[1])
                    wochentag = temp[2]
                    schritte = int(temp[3])
                    array.append([monat, tag, wochentag, schritte])
        except (ValueError, IndexError) as e:
            if dataSet == 1:
                print(f"warnung zeile {i}: {e}")
    
    return array

def clean(array):
    """Bereinigt Datensatz 2 und 3: Korrigiert ungültige Jahre/Monate/Werte
    Regel 1: Jahr außerhalb [1900, 2100] -> aus Nachbarwert ableiten
    Regel 2: Monat = -1 (ungültig) -> Zeile ENTFERNEN
    Regel 3: Wert = 0.0 (ungültig) -> zeitlich nächsten gültigen Wert nutzen"""
    
    # Erst: Entferne Zeilen mit ungültigen Monaten (-1)
    array[:] = [row for row in array if row[1] != -1]
    
    # Dann: Korrigiere verbleibende Fehler
    for i in range(len(array)):
        # Korrigiere ungültiges Jahr
        if not (1900 <= array[i][0] <= 2100):
            if i > 0 and i < len(array) - 1:
                array[i][0] = array[i-1][0]
            elif i > 0:
                array[i][0] = array[i-1][0]
        
        # Korrigiere ungültigen Wert
        if len(array[i]) >= 3 and array[i][2] == 0.0:
            if i > 0 and array[i-1][2] != 0.0:
                array[i][2] = array[i-1][2]
            elif i < len(array) - 1 and array[i+1][2] != 0.0:
                array[i][2] = array[i+1][2]

# ============================================================
# DATEIEXPORT
# ============================================================

def outputToFile(fileName, array, beschreibung):
    """Schreibt vollständige statistische Auswertung in Textdatei
    - Lagemaße: Modus, Mittelwert, Median
    - Streuungsmaße: Spannweite, MAD, Varianz, Variations-Koeffizient
    - Quantile: Quartile (Q1, Q2, Q3, Max), Dezile (D1-D9)"""
    beschreibungen = beschreibung.split(";")
    
    with open(fileName, 'w', encoding='utf-8') as f:
        for spalte_idx in range(len(array[0])):
            # Extrahiere Spalte
            werte = [row[spalte_idx] for row in array]
            
            # Überspringe nicht-numerische Spalten (z.B. Wochentag)
            if werte and isinstance(werte[0], str):
                continue
            
            # Schreibe Statistiken
            f.write(f"{beschreibungen[spalte_idx]}\n")
            f.write(f"Modus: {mode(werte)}\n")
            f.write(f"Mittelwert: {np.mean(werte):.3f}\n")
            f.write(f"Median: {np.median(werte):.3f}\n")
            f.write(f"Spannweite: {max(werte) - min(werte)}\n")
            f.write(f"Abweichung Median: {abweichungMedian(werte):.3f}\n")
            f.write(f"Varianz: {np.var(werte):.3f}\n")
            f.write(f"Variations-Koeff: {variationsKoeffizient(werte):.3f}\n")
            f.write(f"Kovarianz: {np.cov(werte):.3f}\n")
            
            # Schreibe Quartile
            quarts = quartile(werte)
            f.write(f"Q-Abstand: {quarts[2] - quarts[0]}\n")
            f.write(f"Quartile: q25: {quarts[0]}, q50: {quarts[1]}, q75: {quarts[2]}, q100: {quarts[3]}\n")
            
            # Schreibe Dezile
            dezs = dezile(werte)
            dezile_text = ", ".join([f"d{i}: {dezs[i-1]}" for i in range(1, 10)])
            f.write(f"Dezile: {dezile_text}\n\n\n")

def urlist(filePath, fileName, data):
    """Schreibt Urlisten (unsortierte Rohwerte) in separate CSV-Dateien
    Eine Datei pro Spalte: Liste alle Werte in Eingabe-Reihenfolge
    Format: Spaltenname als Header, dann alle Werte untereinander"""
    for spalte_idx in range(len(data[0])):
        ausgabe_pfad = os.path.join(script_dir, filePath + fileName[spalte_idx] + '.csv')
        werte = [str(row[spalte_idx]) for row in data]
        
        with open(ausgabe_pfad, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([fileName[spalte_idx]])
            for wert in werte:
                writer.writerow([wert])
        
def ranglist(filePath, fileName, data):
    """Schreibt Ranglisten (sortierte Werte) in separate CSV-Dateien
    Eine Datei pro Spalte: Liste alle Werte aufsteigend sortiert
    Format: Spaltenname als Header, dann sortierte Werte untereinander"""
    for spalte_idx in range(len(data[0])):
        ausgabe_pfad = os.path.join(script_dir, filePath + fileName[spalte_idx] + '.csv')
        werte = sorted([str(row[spalte_idx]) for row in data])
        
        with open(ausgabe_pfad, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([fileName[spalte_idx]])
            for wert in werte:
                writer.writerow([wert])

# ============================================================
# VISUALISIERUNG
# ============================================================

def boxWhiskerPlot(filePath, fileName, data, yLabel):
    """Erstellt Box-Whisker-Plots (Schachteldiagramme) für jede Spalte
    Zeigt: Min, Q1, Median, Q3, Max + Ausreißer
    Nützlich zur Visualisierung von Verteilung und Streuung"""
    for spalte_idx in range(len(data[0])):
        werte = [row[spalte_idx] for row in data]
        
        # Überspringe nicht-numerische Spalten (z.B. Wochentag)
        if werte and isinstance(werte[0], str):
            continue
        
        # Skaliere große Zahlenbereiche
        if spalte_idx > 1 and ('dataset-1' in filePath or 'dataset-3' in filePath):
            werte = [w / 1000000 for w in werte]
        
        # Zeichne mit Styling
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(werte, patch_artist=True, widths=0.6)
        
        # Farbe und Stil
        for patch in bp['boxes']:
            patch.set_facecolor('#951E2D')
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5, color='#2c2c2c')
        for cap in bp['caps']:
            cap.set(linewidth=1.5, color='#2c2c2c')
        for median in bp['medians']:
            median.set(linewidth=2, color='white')
        
        # Formatierung
        ax.set_title(fileName[spalte_idx], fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(yLabel[spalte_idx], fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Speichere
        ausgabe_pfad = os.path.join(script_dir, filePath + fileName[spalte_idx] + '.jpg')
        plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
        plt.close()

def histogram(data, fileName):
    """Erstellt Histogramm mit Jahresdurchschnitten
    Formel: Durchschnitt_Jahr = (1/n) * Σ Werte_Jahr
    Zeigt Entwicklung über Jahre hinweg als Balkendiagramm"""
    jahresDurchschnitt = {}
    
    # Summiere Werte nach Jahr
    for row in data:
        jahr = row[0]
        wert = row[2]
        if jahr not in jahresDurchschnitt:
            jahresDurchschnitt[jahr] = []
        jahresDurchschnitt[jahr].append(wert)
    
    # Berechne Durchschnitte
    jahre = sorted(jahresDurchschnitt.keys())
    durchschnitte = [sum(jahresDurchschnitt[jahr]) / len(jahresDurchschnitt[jahr]) / 1000000 for jahr in jahre]
    
    # Zeichne mit Styling
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(jahre, durchschnitte, color='#951E2D', alpha=0.8, edgecolor='#2c2c2c', linewidth=1.5)
    
    # Wertlabels auf Balken
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatierung
    ax.set_title('Energieerzeugung - Durchschnitt pro Jahr', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Jahr', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energieerzeugung (TWh)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    
    # X-Achse formatieren
    ax.set_xticks(jahre)
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

def sine_function(x, A, B, C, D):
    """Sinusfunktion für Trendanpassung
    f(x) = A*sin(B*x + C) + D
    A = Amplitude, B = Frequenz (2π für 1 Jahr), C = Phase, D = Vertikalverschiebung"""
    return A * np.sin(B * x + C) + D

def _fit_sine_curve(zeit, werte, initial_guess):
    """Fittet sinusförmige Kurve: f(x) = A*sin(B*x + C) + D
    Parameter: A=Amplitude, B=Frequenz, C=Phase, D=Offset
    Verwendet Least-Squares Fitting mit maximal 5000 Iterationen"""
    try:
        params, _ = curve_fit(sine_function, zeit, werte, p0=initial_guess, maxfev=5000)
        A, B, C, D = params
        print(f"fit: A={A:.3f}, B={B:.3f}, C={C:.3f}, D={D:.3f}")
        return A, B, C, D
    except (RuntimeError, ValueError) as e:
        print(f"fit fehler - verwende initial parameter")
        return initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3]

def scatterPlot(fileName, data, title, yLabel):
    """Erstellt Streudiagramm (Scatter Plot) mit optionalen Trendkurven
    - Datenpunkte als Punkte dargestellt
    - Bei 2 Wertereihen: sinusförmige Trendkurven gefittet
    - Zeigt zeitliche Entwicklung und Saisonalität"""
    if not data:
        return
    
    # Extrahiere Zeit und Werte
    zeit = np.array([row[0] + row[1] / 12 for row in data])
    
    # Skaliere Werte je nach Datensatz
    is_dataset_2 = 'dataset-2' in fileName
    is_large_dataset = 'dataset-1' in fileName or 'dataset-3' in fileName
    werte1 = np.array([row[2] if is_dataset_2 else row[2] / 1000000 for row in data])
    
    # Zeichne erste Wertereihe
    fig, ax = plt.subplots(figsize=(12, 7))
    label1 = 'Ankuenfte' if len(data[0]) > 3 else 'Beschaeftigte'
    ax.scatter(zeit, werte1, color='#951E2D', s=80, alpha=0.7, edgecolors='#2c2c2c', linewidth=1, label=label1, zorder=3)
    
    # Falls zwei Wertereihen: Fittkurven
    if len(data[0]) > 3:
        werte2 = np.array([row[3] / 1000000 for row in data])
        ax.scatter(zeit, werte2, color='#951E2D', s=80, alpha=0.7, edgecolors='#2c2c2c', linewidth=1, label='Uebernachtungen', zorder=3)
        
        # Fitten
        A1, B1, C1, D1 = _fit_sine_curve(zeit, werte1, [4, 6.99, -1991, 7])
        A2, B2, C2, D2 = _fit_sine_curve(zeit, werte2, [-12, 6.74, -1992, 28])
        
        # Zeichne Fittkurven
        fitted1 = sine_function(zeit, A1, B1, C1, D1)
        fitted2 = sine_function(zeit, A2, B2, C2, D2)
        
        ax.plot(zeit, fitted1, color='#E67E22', linewidth=2.5, label=f'Trend 1', alpha=0.8, zorder=2)
        ax.plot(zeit, fitted2, color='#27AE60', linewidth=2.5, label=f'Trend 2', alpha=0.8, zorder=2)
    
    # Formatierung
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Jahr', fontsize=12, fontweight='bold')
    ax.set_ylabel(yLabel, fontsize=12, fontweight='bold')
    ax.set_xlim(np.min(zeit) - 1, np.max(zeit) + 1)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=10)
    
    # Legend
    if len(data[0]) > 3:
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    else:
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Falls Datensatz 2: Zusätzliches Liniendiagramm
    if is_dataset_2:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(zeit, werte1, color='#951E2D', linewidth=2.5, marker='o', markersize=8, 
                label=label1, alpha=0.8, markerfacecolor='#951E2D', markeredgecolor='#2c2c2c', markeredgewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Jahr', fontsize=12, fontweight='bold')
        ax.set_ylabel(yLabel, fontsize=12, fontweight='bold')
        ax.set_xlim(np.min(zeit) - 1, np.max(zeit) + 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        
        line_path = os.path.join(script_dir, fileName.replace('scatterPlot.jpg', 'linePlot.jpg'))
        plt.savefig(line_path, dpi=1147, bbox_inches='tight', facecolor='white')
        plt.close()

def stepsPerWeekday(data, fileName):
    """Erstellt Balkendiagramm mit durchschnittlichen Schritten pro Wochentag
    Zeigt: Durchschnittliche Schritte für jeden Wochentag (Mo-So)"""
    # Wochentag-Mapping
    wochentage_order = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    durchschnitt_wochentag = {tag: [] for tag in wochentage_order}
    
    # Sammle Schritte pro Wochentag
    for row in data:
        wochentag = row[2]
        schritte = row[3]
        if wochentag in durchschnitt_wochentag:
            durchschnitt_wochentag[wochentag].append(schritte)
    
    # Berechne Durchschnitte
    durchschnitte = [sum(durchschnitt_wochentag[tag]) / len(durchschnitt_wochentag[tag]) if durchschnitt_wochentag[tag] else 0 for tag in wochentage_order]
    
    # Zeichne Balkendiagramm
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(wochentage_order, durchschnitte, color='#951E2D', alpha=0.8, edgecolor='#2c2c2c', linewidth=1.5)
    
    # Wertlabels auf Balken
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatierung
    ax.set_title('Durchschnittliche Schritte pro Wochentag', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Wochentag', fontsize=12, fontweight='bold')
    ax.set_ylabel('Schritte', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

def stepsPerMonth(data, fileName):
    """Erstellt Balkendiagramm mit durchschnittlichen Schritten pro Monat
    Zeigt: Durchschnittliche Schritte für jeden Monat (1-12)"""
    durchschnitt_monat = {}
    
    # Sammle Schritte pro Monat
    for row in data:
        monat = row[0]
        schritte = row[3]
        if monat not in durchschnitt_monat:
            durchschnitt_monat[monat] = []
        durchschnitt_monat[monat].append(schritte)
    
    # Berechne Durchschnitte
    monate = sorted(durchschnitt_monat.keys())
    durchschnitte = [sum(durchschnitt_monat[monat]) / len(durchschnitt_monat[monat]) for monat in monate]
    
    # Zeichne Balkendiagramm
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar([intToMonth(m) for m in monate], durchschnitte, color='#951E2D', alpha=0.8, edgecolor='#2c2c2c', linewidth=1.5)
    
    # Wertlabels auf Balken
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatierung
    ax.set_title('Durchschnittliche Schritte pro Monat', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Monat', fontsize=12, fontweight='bold')
    ax.set_ylabel('Schritte', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

def stepsPerDay(data, fileName):
    """Erstellt Liniendiagramm mit durchschnittlichen Schritten pro Monatstag (1-31)
    Zeigt: Durchschnittliche Schritte für jeden Tag des Monats"""
    durchschnitt_tag = {}
    
    # Sammle Schritte pro Monatstag
    for row in data:
        tag = row[1]
        schritte = row[3]
        if tag not in durchschnitt_tag:
            durchschnitt_tag[tag] = []
        durchschnitt_tag[tag].append(schritte)
    
    # Berechne Durchschnitte
    tage = sorted(durchschnitt_tag.keys())
    durchschnitte = [sum(durchschnitt_tag[tag]) / len(durchschnitt_tag[tag]) for tag in tage]
    
    # Zeichne Liniendiagramm
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(tage, durchschnitte, color='#951E2D', linewidth=2.5, marker='o', markersize=6, 
            alpha=0.8, markerfacecolor='#951E2D', markeredgecolor='#2c2c2c', markeredgewidth=1)
    
    # Formatierung
    ax.set_title('Durchschnittliche Schritte pro Monatstag', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Tag des Monats', fontsize=12, fontweight='bold')
    ax.set_ylabel('Schritte', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 32))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

def stepsPerDayBoxPlot(data, fileName):
    """Erstellt Box-Plot für Schritte pro Monatstag (1-31)
    Zeigt: Verteilung der Schritte für jeden Tag des Monats"""
    durchschnitt_tag = {}
    
    # Sammle Schritte pro Monatstag
    for row in data:
        tag = row[1]
        schritte = row[3]
        if tag not in durchschnitt_tag:
            durchschnitt_tag[tag] = []
        durchschnitt_tag[tag].append(schritte)
    
    # Sortiere Tage
    tage = sorted(durchschnitt_tag.keys())
    schritte_pro_tag = [durchschnitt_tag[tag] for tag in tage]
    
    # Zeichne Box-Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(schritte_pro_tag, tick_labels=tage, patch_artist=True, widths=0.6)
    
    # Farbe und Stil
    for patch in bp['boxes']:
        patch.set_facecolor('#951E2D')
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='#2c2c2c')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='#2c2c2c')
    for median in bp['medians']:
        median.set(linewidth=2, color='white')
    
    # Formatierung
    ax.set_title('Verteilung Schritte pro Monatstag', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Tag des Monats', fontsize=12, fontweight='bold')
    ax.set_ylabel('Schritte', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

def stepsPerMonthBoxPlot(data, fileName):
    """Erstellt Box-Plot für Schritte pro Monat (1-12)
    Zeigt: Verteilung der Schritte für jeden Monat"""
    durchschnitt_monat = {}
    
    # Sammle Schritte pro Monat
    for row in data:
        monat = row[0]
        schritte = row[3]
        if monat not in durchschnitt_monat:
            durchschnitt_monat[monat] = []
        durchschnitt_monat[monat].append(schritte)
    
    # Sortiere Monate
    monate = sorted(durchschnitt_monat.keys())
    schritte_pro_monat = [durchschnitt_monat[monat] for monat in monate]
    monatslabels = [intToMonth(m) for m in monate]
    
    # Zeichne Box-Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(schritte_pro_monat, tick_labels=monatslabels, patch_artist=True, widths=0.6)
    
    # Farbe und Stil
    for patch in bp['boxes']:
        patch.set_facecolor('#951E2D')
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='#2c2c2c')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='#2c2c2c')
    for median in bp['medians']:
        median.set(linewidth=2, color='white')
    
    # Formatierung
    ax.set_title('Verteilung Schritte pro Monat', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Monat', fontsize=12, fontweight='bold')
    ax.set_ylabel('Schritte', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Speichere
    ausgabe_pfad = os.path.join(script_dir, fileName)
    plt.savefig(ausgabe_pfad, dpi=1147, bbox_inches='tight', facecolor='white')
    plt.close()

# ==========================================================================
# HAUPTPROGRAMM: DATENANALYSE UND AUSWERTUNG
# ==========================================================================
# Ablauf: 1. Laden aller 4 Datensätze
#         2. Bereinigung und Validierung
#         3. Statistische Auswertung mit Grafiken und Export
# ==========================================================================

# Datenarrays initialisieren
data1Array = []
data2Array = []
data3Array = []
data3bArray = []
data4Array = []

# =========== DATENSATZ 1: ELEKTRIZITÄTSERZEUGUNG (Steinkohle) ===========
# Inhalt: Jahr, Monat, Elektrizitätserzeugung (MWh)
# Quelle: Dataset-1/data-1.csv

print("=" * 60)
print("datensatz 1 laden")
print("=" * 60)

data1_path = os.path.join(script_dir, 'Dataset-1', 'data-1.csv')
print(f"lese: {data1_path}")

try:
    with open(data1_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        print(f"zeilen: {len(lines)}")
        data.seek(0)
        readFromDataToArray(data, data1Array, 1)
    print(f"ok - {len(data1Array)} zeilen geladen")
except FileNotFoundError:
    print(f"fehler: datei nicht gefunden")
    print("überspringe datensatz 1")

if len(data1Array) > 0:
    data1Array.sort()
    print(f"erste: {data1Array[:3]}")
    print(f"letzte: {data1Array[-3:]}")
else:
    print("warnung: keine daten für datensatz 1")

# =============== DATENSATZ 2: BESCHÄFTIGTE (Einzelhandel) ================
# Inhalt: Jahr, Monat, Beschäftigte (% vs 2015)
# Quelle: Dataset-2/data-2.csv

print("\n" + "=" * 60)
print("datensatz 2 laden")
print("=" * 60)

data2_path = os.path.join(script_dir, 'Dataset-2', 'data-2.csv')
print(f"lese: {data2_path}")

try:
    with open(data2_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        print(f"zeilen: {len(lines)}")
        data.seek(0)
        readFromDataToArray(data, data2Array, 2)
    print(f"ok - {len(data2Array)} zeilen geladen")
    
    if len(data2Array) > 0:
        clean(data2Array)
        data2Array.sort()
        print(f"erste: {data2Array[:3]}")
        print(f"letzte: {data2Array[-3:]}")
    else:
        print("warnung: keine daten für datensatz 2")
        
except FileNotFoundError:
    print(f"fehler: datei nicht gefunden")
    print("überspringe datensatz 2")

# ============ DATENSATZ 3: BEHERBERGUNG (Ankünfte + Übernachtungen) ========
# Inhalt: Ort, Ankünfte (Mio), Übernachtungen (Mio)
# Quelle: Dataset-3/data-3-a.csv + data-3-b.csv (kombiniert)

print("\n" + "=" * 60)
print("datensatz 3 laden")
print("=" * 60)

data3a_path = os.path.join(script_dir, 'Dataset-3', 'data-3-a.csv')
print(f"lese teil a: {data3a_path}")

try:
    with open(data3a_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        print(f"zeilen: {len(lines)}")
        data.seek(0)
        readFromDataToArray(data, data3Array, 3)
    print(f"ok - {len(data3Array)} zeilen aus teil a")
except FileNotFoundError:
    print(f"fehler: teil a nicht gefunden")
    print("überspringe teil a")

data3b_path = os.path.join(script_dir, 'Dataset-3', 'data-3-b.csv')
print(f"\nlese teil b: {data3b_path}")

try:
    with open(data3b_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        print(f"zeilen: {len(lines)}")
        data.seek(0)
        readFromDataToArray(data, data3bArray, 31)
    print(f"ok - {len(data3bArray)} zeilen aus teil b")
except FileNotFoundError:
    print(f"fehler: teil b nicht gefunden")
    print("überspringe teil b")

# Beide Teile von Datensatz 3 kombinieren
if len(data3Array) > 0 and len(data3bArray) > 0:
    print("\nkombiniere teil a und b...")
    i = -1
    for teilA in data3Array:
        i += 1
        for teilB in data3bArray:
            if teilA[0] == teilB[0]:
                teilA += teilB[1:]
                del teilA[0]
                break
    
    clean(data3Array)
    data3Array.sort()
    print(f"kombiniert: {len(data3Array)}")
    if len(data3Array) > 0:
        print(f"erste: {data3Array[:3]}")
        print(f"letzte: {data3Array[-3:]}")
else:
    print("warnung: konnte nicht kombinieren")

# ================ DATENSATZ 4: FAHRZEUGMESSUNGEN (Zeit + Geschwindigkeit) ==
# Inhalt: Zeit (Sekunden), Geschwindigkeit (km/h)
# Quelle: Dataset-4/data-4.csv

print("\n" + "=" * 60)
print("datensatz 4 laden")
print("=" * 60)

data4_path = os.path.join(script_dir, 'Dataset-4', 'data-4.csv')
print(f"lese: {data4_path}")

try:
    with open(data4_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        print(f"zeilen: {len(lines)}")
        data.seek(0)
        readFromDataToArray(data, data4Array, 4)
    print(f"ok - {len(data4Array)} zeilen geladen")
    
    if len(data4Array) > 0:
        data4Array.sort()
        print(f"erste: {data4Array[:3]}")
        print(f"letzte: {data4Array[-3:]}")
    else:
        print("warnung: keine daten für datensatz 4")
        
except FileNotFoundError:
    print(f"fehler: datei nicht gefunden")
    print("überspringe datensatz 4")

# ==========================================================================
# AUSWERTUNG: STATISTISCHE ANALYSEN UND GRAFISCHEN DARSTELLUNGEN
# ==========================================================================

print("\n" + "=" * 60)
print("auswertung starten")
print("=" * 60)

# --------- AUSWERTUNG DATENSATZ 1: ELEKTRIZITÄTSERZEUGUNG ---------
# Grafiken: Histogramm (Jahresverlauf), Scatter-Plot, Box-Plots
# Ausgabe: Statistiken (URL/Ranglisten), Deskriptive Statistiken

if len(data1Array) > 0:
    print("\ndatensatz 1 auswertung")
    
    # Ausgabeordner erstellen
    output_dir_1 = os.path.join(script_dir, 'output', 'dataset-1')
    os.makedirs(output_dir_1, exist_ok=True)
    print(f"ordner: {output_dir_1}")
    
    # Histogramm erstellen
    histogram_file = os.path.join('output', 'dataset-1', 'histogramm.jpg')
    histogram(data1Array, histogram_file)
    
    # Urlisten erstellen
    urlist('output/dataset-1/urliste-', ['Jahr', 'Monat', 'Elektrizitaetserzeugung'], data1Array)
    
    # Ranglisten erstellen
    ranglist('output/dataset-1/rangliste-', ['Jahr', 'Monat', 'Elektrizitaetserzeugung'], data1Array)
    
    # Box-Whisker-Plots erstellen
    boxWhiskerPlot('output/dataset-1/boxWhiskerPlot-', ['Jahr', 'Monat', 'Elektrizizaetserzeugung'], data1Array, ['', '', 'in TWh'])
    
    # Scatter-Plot erstellen
    scatterPlot('output/dataset-1/scatterPlot.jpg', data1Array, 'Energieerzeugung aus Steinkohle nach Jahren', 'Energieerzeugung in TWh')
    
    # Statistische Auswertung
    output_file = os.path.join(script_dir, 'output', 'dataset-1', 'content.txt')
    outputToFile(output_file, data1Array, "Variable Jahr: ;Variable Monat: ;Variable Elektrizizaetserzeugung netto in MWh: ")
    
    print("datensatz 1 fertig")
else:
    print("\ndatensatz 1 übersprungen")

# ---------- AUSWERTUNG DATENSATZ 2: BESCHÄFTIGTE (Einzelhandel) ----------
# Grafiken: Scatter-Plot, Liniendiagramm, Box-Plots
# Ausgabe: Statistiken (URL/Ranglisten), Deskriptive Statistiken

if len(data2Array) > 0:
    print("\ndatensatz 2 auswertung")
    
    # Ausgabeordner erstellen
    output_dir_2 = os.path.join(script_dir, 'output', 'dataset-2')
    os.makedirs(output_dir_2, exist_ok=True)
    print(f"ordner: {output_dir_2}")
    
    # Urlisten erstellen
    urlist('output/dataset-2/urliste-', ['Jahr', 'Monat', 'Beschaeftigte'], data2Array)
    
    # Ranglisten erstellen
    ranglist('output/dataset-2/rangliste-', ['Jahr', 'Monat', 'Beschaeftigte'], data2Array)
    
    # Box-Whisker-Plots erstellen
    boxWhiskerPlot('output/dataset-2/boxWhiskerPlot-', ['Jahr', 'Monat', 'Beschäftigte'], data2Array, ['', '', 'prozentual zu 2015'])
    
    # Scatter-Plot erstellen
    scatterPlot('output/dataset-2/scatterPlot.jpg', data2Array, 'Beschäftige im Einzelhandel im Vergleich zu 2015 nach Jahren', 'Beschaeftigte prozentual zu 2015')
    
    # Statistische Auswertung
    output_file = os.path.join(script_dir, 'output', 'dataset-2', 'content.txt')
    outputToFile(output_file, data2Array, "Variable Jahr: ;Variable Monat: ;Variable Beschaeftigte prozentual zu 2015: ")
    
    print("datensatz 2 fertig")
else:
    print("\ndatensatz 2 übersprungen")

# ------- AUSWERTUNG DATENSATZ 3: BEHERBERGUNG (Ank./Übern.) -------
# Grafiken: Scatter-Plot mit Fittkurven, Box-Plots
# Ausgabe: Konsolidierte CSV, Statistiken (URL/Ranglisten)

if len(data3Array) > 0 and len(data3Array[0]) > 3:
    print("\ndatensatz 3 auswertung")
    
    # Ausgabeordner erstellen
    output_dir_3 = os.path.join(script_dir, 'output', 'dataset-3')
    os.makedirs(output_dir_3, exist_ok=True)
    print(f"ordner: {output_dir_3}")
    
    # Urlisten erstellen
    urlist('output/dataset-3/urliste-', ['Jahr', 'Monat', 'Ankuenfte', 'Uebernachtungen'], data3Array)
    
    # Ranglisten erstellen
    ranglist('output/dataset-3/rangliste-', ['Jahr', 'Monat', 'Ankuenfte', 'Uebernachtungen'], data3Array)
    
    # Box-Whisker-Plots erstellen
    boxWhiskerPlot('output/dataset-3/boxWhiskerPlot-', ['Jahr', 'Monat', 'AnzahlAnkuenfte', 'AnzahlUebernachtungen'], data3Array, ['', '', 'in Mio', 'in Mio'])
    
    # Scatter-Plot erstellen
    scatterPlot('output/dataset-3/scatterPlot.jpg', data3Array, 'Ankuenfte und Uebernachtungen in Beherbergungsbetrieben nach Jahren', 'Anzahl in Mio')
    
    # Statistische Auswertung
    output_file = os.path.join(script_dir, 'output', 'dataset-3', 'content.txt')
    outputToFile(output_file, data3Array, "Variable Jahr: ;Variable Monat: ;Variable Anzahl an Unterkünften: ;Variable Anzahl an Uebernachtugnen: ")
    
    # Konsolidierte CSV-Datei erstellen
    konsolidiert_file = os.path.join(script_dir, 'output', 'dataset-3', 'konsolidiert.csv')
    with open(konsolidiert_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for content in data3Array:
            writer.writerow([str(content[0]), intToMonth(content[1]), str(content[2]), str(content[3])])
    
    print("datensatz 3 fertig")
else:
    print("\ndatensatz 3 übersprungen")

# --------- AUSWERTUNG DATENSATZ 4: FAHRZEUGMESSUNGEN ---------
# Grafiken: Box-Plots (Zeit + Geschwindigkeit)
# Ausgabe: Statistiken (URL/Ranglisten), Deskriptive Statistiken

if len(data4Array) > 0:
    print("\ndatensatz 4 auswertung")
    
    # Ausgabeordner erstellen
    output_dir_4 = os.path.join(script_dir, 'output', 'dataset-4')
    os.makedirs(output_dir_4, exist_ok=True)
    print(f"ordner: {output_dir_4}")
    
    # Urlisten erstellen
    urlist('output/dataset-4/urliste-', ['Monat', 'Tag', 'Wochentag', 'Anzahl Schritte'], data4Array)
    
    # Ranglisten erstellen
    ranglist('output/dataset-4/rangliste-', ['Monat', 'Tag', 'Wochentag', 'Anzahl Schritte'], data4Array)
    
    # Box-Whisker-Plots erstellen
    boxWhiskerPlot('output/dataset-4/boxWhiskerPlot-', ['Monat', 'Tag', 'Wochentag', 'Anzahl Schritte'], data4Array, ['', '', '', 'Schritte'])

    # Box-Plots für durchschnittliche Schritte
    stepsPerDayBoxPlot(data4Array, 'output/dataset-4/boxPlot-Tag.jpg')
    stepsPerMonthBoxPlot(data4Array, 'output/dataset-4/boxPlot-Monat.jpg')

    # Spezielle Grafiken für Schritte-Analyse
    stepsPerWeekday(data4Array, 'output/dataset-4/durchschnittSchritte-Wochentag.jpg')
    stepsPerMonth(data4Array, 'output/dataset-4/durchschnittSchritte-Monat.jpg')
    stepsPerDay(data4Array, 'output/dataset-4/durchschnittSchritte-Tag.jpg')

    # Statistische Auswertung
    output_file = os.path.join(script_dir, 'output', 'dataset-4', 'content.txt')
    outputToFile(output_file, data4Array, "Variable Monat: ;Variable Tag: ;Variable Wochentag: ;Variable Anzahl Schritte: ")

    print("datensatz 4 fertig")
else:
    print("\ndatensatz 4 übersprungen")

print("\n" + "=" * 60)
print("fertig")
print("=" * 60)

print("\nzusammenfassung:")
print("-" * 40)
print(f"d1: {len(data1Array)} zeilen")
print(f"d2: {len(data2Array)} zeilen")
print(f"d3: {len(data3Array)} zeilen")
print(f"d4: {len(data4Array)} zeilen")
print("-" * 40)
print(f"ergebnisse: {os.path.join(script_dir, 'output')}")
print("=" * 60)
