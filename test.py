
import requests
import numpy as np
from goprocam import GoProCamera, constants
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import time

from pynput import keyboard

# Funkcja wywoływana po wciśnięciu klawisza
def on_key_press(key):
    try:
        if key == keyboard.Key.enter:
            # Tutaj możesz umieścić kod do wykonania po wciśnięciu Enter
            print("Wciśnięto Enter")
    except AttributeError:
        pass

# Nasłuchiwanie na wciśnięcia klawiszy
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

# Program będzie działał i nasłuchiwał na wciśnięcie


def extract_card_labels(sorted_predictions):
    card_labels = []
    for label, _ in sorted_predictions:
        # Usuwanie nawiasów kwadratowych i apostrofów
        clean_label = label.strip("[]'")
        card_labels.append(clean_label)
    return card_labels


def get_card_positions_by_suit(card_labels):
    # Inicjalizacja słowników dla każdego koloru
    suits_positions = {'hearts': [], 'diamonds': [], 'clubs': [], 'spades': []}

    # Iteracja przez listę nazw kart i zapisywanie ich pozycji
    for i, card_label in enumerate(card_labels):
        if 'hearts' in card_label:
            suits_positions['hearts'].append(i + 1)
        elif 'diamonds' in card_label:
            suits_positions['diamonds'].append(i + 1)
        elif 'clubs' in card_label:
            suits_positions['clubs'].append(i + 1)
        elif 'spades' in card_label:
            suits_positions['spades'].append(i + 1)

    return suits_positions
# Wywołanie funkcji dla przykładowej listy sorted_predictions


def announcement(komunikat):
    print(komunikat)
    os.system(f"say  '{komunikat}'")
def sort_cards(lista_kart):
    kolejnosc_kolorow = ['spades', 'hearts', 'diamonds', 'clubs']
    polskie_nazwy_kolorow = {'spades': 'pik', 'hearts': 'kier', 'diamonds': 'karo', 'clubs': 'trefl'}
    wartosci_kart = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    for kolor in kolejnosc_kolorow:
        for wartosc in sorted(wartosci_kart.values()):
            for i in range(len(lista_kart)):
                if lista_kart[i].endswith(kolor) and wartosci_kart[lista_kart[i][0]] == wartosc:
                    karta_do_przeniesienia = lista_kart[i]
                    pozycja_karty = i + 1  # pozycja liczona od lewej, od 1
                    pozycja_docelowa = 1  # pozycja na początku listy

                    kolor_karty = polskie_nazwy_kolorow[kolor]
                    if 'spades' in karta_do_przeniesienia:
                        karta_do_przeniesienia = karta_do_przeniesienia.replace('spades', kolor_karty)
                    elif 'hearts' in karta_do_przeniesienia:
                        karta_do_przeniesienia = karta_do_przeniesienia.replace('hearts', kolor_karty)
                    elif 'diamonds' in karta_do_przeniesienia:
                        karta_do_przeniesienia = karta_do_przeniesienia.replace('diamonds', kolor_karty)
                    elif 'clubs' in karta_do_przeniesienia:
                        karta_do_przeniesienia = karta_do_przeniesienia.replace('clubs', kolor_karty)

                    announcement(
                        f"Przenieś kartę {karta_do_przeniesienia} z pozycji {pozycja_karty} na pozycję {pozycja_docelowa} ")

                    lista_kart.insert(pozycja_docelowa - 1, lista_kart.pop(i))
                    time.sleep(5)  # 5-sekundowe opóźnienie
                    break
    return lista_kart
def announce_card_positions(suits_positions):
    # Słownik do zamiany angielskich nazw kolorów na polskie
    polish_suit_names = {'spades': 'piki', 'hearts': 'kiera', 'diamonds': 'karo', 'clubs': 'trefle'}

    for suit, positions in suits_positions.items():
        # Zamiana nazwy koloru na polski
        polish_suit = polish_suit_names.get(suit, suit)

        if positions:
            positions_str = ', '.join(map(str, positions))
            announcement = f"Karty {polish_suit} są na pozycjach: {positions_str}"
        else:
            announcement = f"Brak kart {polish_suit}"

        print(announcement)
        # Odczytywanie komunikatu na głos, zależne od systemu i dostępnych narzędzi
        os.system(f"say -v Zosia '{announcement}'")





def get_photo_url(photo_number):
    return f"http://10.5.5.9:8080/videos/DCIM/100GOPRO/GOPR{photo_number:04d}.JPG"

LAST_PHOTO_NUMBER_FILE = "/Users/bartosz/Desktop/frames/last.txt"

def save_last_photo_number(photo_number):
    try:
        with open(LAST_PHOTO_NUMBER_FILE, "w") as file:
            file.write(str(photo_number))
    except IOError as e:
        print(f"Błąd przy zapisie do pliku: {e}")

def load_last_photo_number():
    if os.path.exists(LAST_PHOTO_NUMBER_FILE):
        try:
            with open(LAST_PHOTO_NUMBER_FILE, "r") as file:
                return int(file.read().strip())
        except IOError as e:
            print(f"Błąd przy odczycie z pliku: {e}")
            return 324  # Domyślny początkowy numer zdjęcia, jeśli wystąpił problem
    else:
        return 324  # Domyślny początkowy numer zdjęcia, jeśli plik nie istnieje

last_photo_number = load_last_photo_number()

def get_latest_photo():
    global last_photo_number
    photo_url = f"http://10.5.5.9:8080/videos/DCIM/100GOPRO/GOPR{last_photo_number:04d}.JPG"
    response = requests.get(photo_url)
    if response.status_code == 200:
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        last_photo_number += 1  # Zwiększ numer zdjęcia
        save_last_photo_number(last_photo_number)  # Zapisz nowy numer zdjęcia
        return image
    else:
        print(f"Nie można pobrać zdjęcia z adresu: {photo_url}")
        return None


# Funkcje pomocnicze
def draw_text(img, text, position, font_scale=1, font_thickness=2, text_color=(255, 255, 255)):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def is_significantly_lower(box, average_y, threshold=800):
    return box[1] > (average_y + threshold)

# Funkcje do obsługi kamery GoPro
def connect_to_gopro():
    gpCam = GoProCamera.GoPro()
    gpCam.overview()
    return gpCam

def take_photo(gpCam):
    gpCam.take_photo(3)  # 3 sekundy opóźnienia
    return gpCam.getMedia()

def download_photo(gpCam, media_url):
    # Pobranie nazwy pliku i folderu z URL
    media_folder = media_url.split('/')[-2]
    file_name = media_url.split('/')[-1]
    save_path = f"/Users/bartosz/Desktop/frames/{file_name}"

    # Pobranie i zapisanie mediów
    gpCam.downloadMedia(media_folder, file_name, custom_filename=save_path)

    return save_path

# Funkcja do analizy kart
def predict_cards(model, img):
    results = model.predict(img)
    cards = {}
    for box in sorted(results[0].boxes, key=lambda x: x.xyxy[0][0]):
        b = box.xyxy[0]
        c = box.cls
        confidence = box.conf.item()
        if confidence < 0.3:
            continue
        label = model.names[int(c)]
        if label not in cards or cards[label][1] < confidence:
            cards[label] = (b, confidence)
    return cards

def select_best_card(existing_cards, label, box, confidence):
    for existing_label, (existing_box, existing_confidence) in list(existing_cards.items()):
        if calculate_iou(existing_box, box) > 0.8:
            if confidence > existing_confidence:
                del existing_cards[existing_label]
                existing_cards[label] = (box, confidence)
            return
    existing_cards[label] = (box, confidence)

# Funkcja do odczytywania kart na głos
def read_cards_aloud(sorted_predictions):
    # Słownik do zamiany angielskich nazw kolorów na polskie
    polskie_nazwy_kolorow = {'spades': 'pik', 'hearts': 'kier', 'diamonds': 'karo', 'clubs': 'trefl'}

    # Przetwarzanie etykiet kart i zamiana nazw kolorów na polskie
    translated_labels = []
    for label in sorted_predictions:
        # Wyszukanie koloru w etykiecie karty
        for english_suit, polish_suit in polskie_nazwy_kolorow.items():
            if english_suit in label:
                # Zamiana angielskiej nazwy koloru na polską
                translated_label = label.replace(english_suit, polish_suit)
                translated_labels.append(translated_label)
                break  # Przerywamy pętlę po znalezieniu i zamianie koloru

    # Tworzenie tekstu do odczytania
    text = ', '.join(translated_labels)

    # Odczytywanie kart na głos, przykład dla systemu MacOS z głosem Zosia
    os.system(f"say -v Zosia '{text}'")


def main():
    gpCam = connect_to_gopro()
    model_paths = [
        '/Users/bartosz/Desktop/najlepszemodele/czyjestZmiana2.pt',
        '/Users/bartosz/Desktop/najlepszemodele/TOPTOP2.pt',
    ]
    models = [YOLO(path) for path in model_paths]
    sorted_cards = []
    def on_key_press(key):
        nonlocal sorted_cards
        if key == keyboard.Key.esc:
            print("Wyjście z programu.")
            return False  # Zakończ nasłuchiwanie na wciśnięcia klawiszy

        if key == keyboard.Key.enter:
            take_photo(gpCam)
            time.sleep(5)  # Oczekiwanie, aby kamera zdążyła zrobić zdjęcie

            img = get_latest_photo()
            if img is None:
                return True  # Kontynuuj nasłuchiwanie na klawisze

            final_predictions = {}
            for model in models:
                predictions = predict_cards(model, img)
                for label, (b, confidence) in predictions.items():
                    select_best_card(final_predictions, label, b, confidence)

            average_y = sum([b[1] for _, (b, _) in final_predictions.items()]) / len(final_predictions)
            sorted_predictions = sorted(
                [(label, (b, confidence)) for label, (b, confidence) in final_predictions.items() if not is_significantly_lower(b, average_y)],
                key=lambda item: item[1][0][0]
            )
            sorted_predictions.extend(
                [(label, (b, confidence)) for label, (b, confidence) in final_predictions.items() if is_significantly_lower(b, average_y)]
            )

            annotator = Annotator(img)
            for label, (b, confidence) in sorted_predictions:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
                draw_text(img, f'{label}: {confidence:.2f}', (int(b[0]), int(b[1]) - 10), font_scale=1, text_color=(255, 255, 255), font_thickness=3)

            flattened_predictions = [label[0] for label in sorted_predictions if label]
            read_cards_aloud(flattened_predictions)  # Odczyt wyników na głos

            # Generowanie i wyświetlanie instrukcji dotyczących przestawiania kart
            extracted_labels = extract_card_labels(sorted_predictions)
            suits_positions = get_card_positions_by_suit(extracted_labels)
            print(extracted_labels)
            announce_card_positions(suits_positions)
            sorted_cards = sort_cards(extracted_labels)
            read_cards_aloud(sorted_cards)
            print(sorted_cards)

        if key == keyboard.Key.space:
            # Odczytaj karty na głos po wciśnięciu spacji
            if sorted_cards:
                print("Odczytuję karty na głos:")
                read_cards_aloud(sorted_cards)
            else:
                print("Brak posortowanych kart do odczytania na głos.")
    # Nasłuchiwanie na wciśnięcia klawiszy Enter i Esc
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    # Oczekiwanie na wciśnięcie klawisza Esc, aż użytkownik zdecyduje się wyjść
    listener.join()

if __name__ == "__main__":
    main()


