# dw-car-price-prediction
## Readme v1

Zadanie konkursowe na https://www.kaggle.com/c/dw-car-price-prediction/data

Plików jest dużo, ale postaram się to wyjaśnić. Zwrócę uwagę na co ciekawsze notebooki: 

## FE znajduje się w notebookach feature_engineering_....
1. feature_engineering_1__created_at__pln_to_euro_log__geo_lat_long.ipynb: 
Na począku starałem się przeliczyć kurs EUR na PLN. Brałem tygodniowe średnie kursy. W modelowaniu brałem kurs przeliczony, następnie zlogarytmowany i - 10, ponieważ zauważyłem, że srodek rozkałady był na +10. Potem wycofałem się z tego, ponieważ wyniki okazały się gorsze niż ze startera od Vladimira. 
Spróbowałem też zamienić adres na wspołrzędne geograficzne. Zmienne wchodziły do modeli, ale nie były tak silne jak województwo / miasto. Adresy poprawiałem ręcznie sprawdzając z internetu. 

2. feature_engineering_1__created_at__pln_to_euro_log__geo_lat_long.ipynb: tutaj połączylem cechy PL i ANG. Ponadto na podstawie wikipedii lączyłem typy nadwozi albo enkodowałem kraje pochodzenia starając się aby sąsiednie  albo "podobne" kraje miały zbliżone ID. 

2. w feature_engineering_3__MCA.ipynb probowałem analizy korespondencji żeby zmienne binarne feature_* zmienić na mniejszą liczbę wymiarow. Raczej nie przydatne. 

3. Kolejne notebooki to zwykły FE. Tu uwaga: Pythona się dopierop uczę, więc wybaczcie niedociągniecia. 

4. feature_engineering_6 - tu ulepszyłem trochę ekstrakcję województw, ponieważ uwzgęldniłęm 45 największych miast w Polsce. 


## Teraz warto myślę zapoznać się z plikami mozo2.py i mozo.py. To helpery. Podana kolejność wynika z tego, że plik 2 jest bardziej dojrzaly. 
- train_and_submit : zapisuje model w wybranym katalogu, z nazwa i opisem.
Jesli ustawisz param learning_curve, to ponadto wrzuci wykres do pliku. 
* kaggle_min - automatycznie wrzuci na kaggle - w opisie liczba zmiennych i same zmienne posortowane alfabetycznie + hiperparametry i opcjonalny opis.  
* add_model_column_min - eksperymentowalem z modelami od progonozy na 70% danych, mozna od razu zapisac to jako kolumne danych. Ostatecznie sprowadzilo to klopoty, bo przez przypadek (recznie) zapisalem kolumne z prognoza na calym zbiorze i modele potem byly przeuczone. 

- print_plots_by_type : ciekawe wykresy ze wzgl na cene, mozna ogl w notebokach

mozo.py:
- mutate_rand_feature - o tym potem

- merge_with_features - bylo przydatne jak osobno zapisywalem train i test, potem od tego odszdlem, Pozwalalo wczytywac tylko nowo ododane kolumny z CSV (w pliku wynikowym bylo  tylko car_id i nowe zmienne). 

 - num_to_range_categories : zmienna numeryczna na biny 
 
 
## Modelowanie 
Mozecie pominąć pliki modelowanie_* i przejść do modelowane_nowy_*. 
1. modelowane_nowy_start_1.ipynb - tutaj jest więcej FE, ponieważ wyniki FE  w 1 tygodniu byly nie najlepsze. W poprzednich notebookach wszędzie braki danych zastępowałem mediana, Teraz chciałem wyprobować -1. 

2. Przyjalem zadade: modeluje na 70% , oceniam na 30%. Wysylam na kaggla na 100%. Podzial byl ze stratyfikacja na percentylach price_value i kategorii sprzedawcy. 

3. Samo modelowanie. Bylo wiele prób, więc wskażę tylko ciekawsze reprezntatywne przyklady: 
- modelowane_nowy_start_9_selekcja : użylem xgbfir. Pondato uzywalem waznosci z RandomForrest. 
- modelowane_nowy_start_11_xgb_hiperopt : po prostu hiperoptumalizacja wybranego modelu
- modelowane_nowy_start_18_xgb_sprawdzenie_outlierow_Lcurv: przykladowa krzywa uczenia już zoptymalizowanego modelu . Tu po hiperopcie, ponieważ chcialem sprawdzić model przed wrzuceniem na kaggla, Zwykla sciezka to najpierw znalezienie modelu, krzywa, hiperopt, trenoowanie na calym train, 

4. Zainspirowalem sie webinarem który polecal Vladimir. Bylko tam o losowym szukaniu cech. Napisalem kod który dla modelu z ok. 50, 70 i 100 zmiennymi losowal parę zmienych. Jeśli wynik nie pogorszyl się o deltę = 500  do 1500 (rożne proby), to zamienialem cechy. Zapisywałem zmiany i w jednej pętli nie mogly się powrorzyć (uporządkowane pary A -> B). Ale nad tym byla jeszcze jedna pętla.
Jeśli zamiana była na lepsze, to nazywalem to Improvement, Jesli zamiana była na nieco gorsze , to nazywałem to Mutacją (w analogii do ewoliucji nie każda zmiana jest na lepsze, ale bez takich zamian na chwilowo na gorsze nie byloby ludzkości xD ). 
Zwykle ten automat generowal coraz lepsze modele przez kila godzin. Było 50 - 90 estymatorów, więc każda proba to kilka sekund. Gdy nie bylo popraw przez kilka godzin, eksperyment przerywalłm i zaczynałem z innym modelem startowym i ilością zmennych (modele wybrane ręcznie z ważnosci cech). Funkcja train_and_submit automatycznie zapisywala wynik przy polepszeniu (nie mutacji), więc nawet jak serwer padł to wiedzialem od czego zacząć. Puszcałem to głownie w nocy, kiedy i tak spałem xD
Przyklad: modelowane_nowy_start_14_xgb_search_parami_delta

5. Zainspriowalem się też regresją krokową forward i backward. Tylko forward od modelu z 50 zmiennymi dzialal. modelowane_nowy_start_15_xgb_search_backward.ipynb i modelowane_nowy_start_15_xgb_search_forward.ipynb. 

Ogolnie eksperymenty w 4 i 5 nie dawaly lepszych modeli niż najlepszy model na 120 zmiennych, ale wyniki byly bliskie, a liczyły sie na ~500 i ~900 estymatorach a nie na ~2000 więc życiowo były lepsze do pracy. Wiem to już po konkursie bo na skutek blędu jedna ze zmiennych była kolumna z modelem liczonym na calym train, co prowadziolo do trudnosci z oszacowaniem ktory model jest lepszy. Po jej usunieciu znalezione modele były OK.  
Zauwazłylem to po konkursie. 

## Pora na najlepszy model, Jest dość przypadkowy, zanim jeszcze zacząłem na dobre sprawdzanie waznosci cech. Dzieki temu ża na kaggle mam pełną historię w opisie dzięki uploadowi z kodu, to mogę szybko wkleić parametry mojego modelu: 

plik: mae_2188_48244_r2_0_98573_XGBRegressor_start_vars_train_100.csv
{ "mean_absolute_error": 2188.4824444174888, "r2_score": 0.9857258975593151, "vars": "'feature_abs', 'feature_alarm', 'feature_alufelgi', 'feature_asr__kontrola_trakcji_', 'feature_asystent_parkowania', 'feature_asystent_pasa_ruchu', 'feature_bluetooth', 'feature_cd', 'feature_centralny_zamek', 'feature_czujnik_deszczu', 'feature_czujnik_martwego_pola', 'feature_czujnik_zmierzchu', 'feature_czujniki_parkowania_przednie', 'feature_czujniki_parkowania_tylne', 'feature_dach_panoramiczny', 'feature_elektrochromatyczne_lusterka_boczne', 'feature_elektrochromatyczne_lusterko_wsteczne', 'feature_elektryczne_szyby_przednie', 'feature_elektryczne_szyby_tylne', 'feature_elektrycznie_ustawiane_fotele', 'feature_elektrycznie_ustawiane_lusterka', 'feature_esp__stabilizacja_toru_jazdy_', 'feature_gniazdo_aux', 'feature_gniazdo_sd', 'feature_gniazdo_usb', 'feature_hak', 'feature_hud__wyświetlacz_przezierny_', 'feature_immobilizer', 'feature_isofix', 'feature_kamera_cofania', 'feature_klimatyzacja_automatyczna', 'feature_klimatyzacja_czterostrefowa', 'feature_klimatyzacja_dwustrefowa', 'feature_klimatyzacja_manualna', 'feature_komputer_pokładowy', 'feature_kurtyny_powietrzne', 'feature_mp3', 'feature_nawigacja_gps', 'feature_odtwarzacz_dvd', 'feature_ogranicznik_prędkości', 'feature_ogrzewanie_postojowe', 'feature_podgrzewana_przednia_szyba', 'feature_podgrzewane_lusterka_boczne', 'feature_podgrzewane_przednie_siedzenia', 'feature_podgrzewane_tylne_siedzenia', 'feature_poduszka_powietrzna_chroniąca_kolana', 'feature_poduszka_powietrzna_kierowcy', 'feature_poduszka_powietrzna_pasażera', 'feature_poduszki_boczne_przednie', 'feature_poduszki_boczne_tylne', 'feature_przyciemniane_szyby', 'feature_radio_fabryczne', 'feature_radio_niefabryczne', 'feature_regulowane_zawieszenie', 'feature_relingi_dachowe', 'feature_system_start_stop', 'feature_szyberdach', 'feature_tapicerka_skórzana', 'feature_tapicerka_welurowa', 'feature_tempomat', 'feature_tempomat_aktywny', 'feature_tuner_tv', 'feature_wielofunkcyjna_kierownica', 'feature_wspomaganie_kierownicy', 'feature_zmieniarka_cd', 'feature_łopatki_zmiany_biegów', 'feature_światła_do_jazdy_dziennej', 'feature_światła_led', 'feature_światła_przeciwmgielne', 'feature_światła_xenonowe', 'new_param_emisja_co2_cat', 'new_param_kategoria_cat', 'new_param_kod_silnika_cat', 'new_param_kolor_cat', 'new_param_kraj_pochodzenia_cat', 'new_param_liczba_pozostałych_rat_cat', 'new_param_marka_pojazdu_cat', 'new_param_miesięczna_rata_cat', 'new_param_model_pojazdu_cat', 'new_param_napęd_cat', 'new_param_oferta_od_cat', 'new_param_opłata_początkowa_cat', 'new_param_pierwsza_rejestracja_cat', 'new_param_rodzaj_paliwa_cat', 'new_param_skrzynia_biegów_cat', 'new_param_stan_cat', 'new_param_typ_cat', 'new_param_vin_cat', 'new_param_wartość_wykupu_cat', 'new_param_wersja_cat', 'new_seller_address_cat', 'new_seller_name_cat', 'new_seller_type_cat', 'param_akryl__niemetalizowany_', 'param_bezwypadkowy', 'param_faktura_vat', 'param_filtr_cząstek_stałych', 'param_homologacja_ciężarowa', 'param_kierownica_po_prawej__anglik_', 'param_leasing', 'param_liczba_drzwi', 'param_liczba_miejsc_2', 'param_matowy', 'param_metalik', 'param_moc', 'param_możliwość_finansowania', 'param_perłowy', 'param_pierwsza_rejestracja_date_na', 'param_pierwszy_właściciel', 'param_pojemność_skokowa', 'param_przebieg', 'param_rok_produkcji', 'param_serwisowany_w_aso', 'param_tuning', 'param_uszkodzony', 'param_vat_discount', 'param_vat_free', 'param_vat_marża', 'param_zarejestrowany_jako_zabytek', 'param_zarejestrowany_w_polsce', 'price_currency_pln'", "best_params": { "n_estimators": 1879, "learning_rate": 0.017354274677292683, "subsample": 0.923403370465418, "colsample_bytree": 0.8797393544001986, "max_depth": 43, "min_child_weight": 48, "reg_alpha": 1.1048264246167496, "reg_lambda": 0.7451037079565321, "random_state": 6950 }, "model_start": "16:25:24", "model_end": "16:38:38", "opis": "" }
priv score 5637.91433
public score 5466.72056

## Uwagi końcowe:
- Ciekawe że już po konkursie przeliczylem go na n_estimators": 3879  i public score = 5424.56438. Model byl z piątku i na serwerze było dużo osób, nie chciałem liczyć tak dużego modelu - ten ~4 tys liczyl się godzinę i 20 min (!).  
- Można używać jak ofeatura inny model (np CatBoost czy lasy), o ile sam jest prognozwany na nie calym zbiorze (np train_70), a jako zmienna wchodzi jako predykcja dla total = train_70 + train_30 + test). Ważny jest dobry podzial train na 2 części
- Nie dzialalo u mnie ELIC ani zmiana fukcji score w xgboost
- Nie widziałem poprawy po usunięciu outlierow w train_70
- Nie zdążylem opracować powtórzeń ogłoszen (niektóre byly ponawiane)
- Mialem jeszcze pomysl z predykcją binu, w ktortym powinna być cena, żeby ograniczyć zbyt niskie / zbyt wysokie predykcje. Nie wiem, czy to w ogóle dobry kierunek. 
- Warto szybko iterować i wrzucać proby na kaggla
- Zdaje sobie sprawę że model pracy był prymitywny wręcz brutalny, ale dużo się nauczyłem wyciągając wnioski po. Znalazlem bład z przypadkowym modelem. Nie jestem programistą Pythona więc też cenne doświadczenie. 



