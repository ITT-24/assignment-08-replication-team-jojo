# Documentation
## Paper auswahl
Die meisten Papers, die uns bei der Recherche begegneten waren innerhalb zwei Wochen nicht umsetzbar oder benötigten zu viel spezielle Technologien/Hardware.
Neben dem Paper zu "Augmented Letters" kamen noch "EyeBookmark"- Ein mit hilfe von eye-tracking automatischer bookmarker von Text -, "DriftBoard"- Eine touch-basierte lösung für Tastaturen auf kleinen Geräten wie Smartwatches - und "LightAnchors"- Eine mit Lichtpunkten funktionierende räumliche verankerung von Daten.


## Implementierung
Wir entschieden uns, die Technick der "Augmented Letters" um eine Gesten-basierte Steuerung zu erweitern, die, mit hilfe von Handtracking, verschiedene Buchstaben erkennen, und befehle ausführen kann. Die Befehle können vom Nutzer individuell in einer "commands.json"-Datei eingestellt werden.

### Graphische Darstellung
Um dem Nutzer ein Visuelles Feedback seiner Eingabe zu geben wird über den gesamten Bildschirm ein unsichtbares Fenster gelegt. Auf diesem wird dann, sobald eine Geste erkannt wird, eine Linie gezeichnet. Sobald der Nutzer fertig mit Zeichnen ist, und sich nicht mehr bewegt, wird der gezeichnete Buchstabe analysiert, und entsprechend ein Menü mit den korrekten Befehlen geöffnet.