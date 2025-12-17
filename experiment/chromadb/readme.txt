# Algemene filosofie #
Het beste zou zijn om de database ergens live te hosten maar als tussenvorm heb ik nu even google drive gekozen.
Om iedereen met dezelfde database te kunnen laten werken is het script zo gemaakt dat als je de database van de drive haalt in dezelfde map zet zoals in de config.ini-file hij de database automatisch ziet.
Mocht jezelf aanpassingen gedaan hebben, even zippen en op dezelfde plaats in drive zetten zodat iedereen de laatste heeft.


# Stappen #
0. Maak virtuele python omgeving
1. Installeer de python requirements via 'pip install -r requirements.txt'
2. Google sheets: deze staat al goed in config.ini
3. Maak in de virtuele omgeving de map aan voor de database (voor naam zie config.ini of verzin zelf naam en pas config.ini aan)
4. Kopieer de laatste database en unzip naar deze map, of database opbouwen via script: 1_importeer_data.py
5. Testen en visualiseren van database via: 2_check_en_visualiseer.py
6. Als je aanpassingen gedaan hebt, zip de laatste database en zet op de drive in dezelfde map als waar je de laatste vandaan hebt met tijd + datum in de naam