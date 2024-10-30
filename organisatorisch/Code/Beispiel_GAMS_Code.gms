Sets
j        Jobs
r        Ressourcen
t        Perioden
s        Schritte
SJ(s, j) Schritte des Jobs j
;

alias(t, tau);

Parameter
a(j, s, r) benötigte Kapazitätseinheiten der Ressource r für Schritt s des Jobs j
b(r, t)   Kapazität der Ressource r in Periode t
c         Umrechnungsfaktor von Perioden in Stunden
d(j, s)   Dauer von Schritt s des Jobs j
ls(j)     letzter Schritt von Job j
;

Positive Variables
TF(j, s) Fertigstellungszeitpunkt von Schritt s des Jobs j
;

Binary Variables
X(j, s, t) binäre Variable mit Wert 1 wenn in Periode t der Schritt s des Jobs j beendet wird 0 sonst
;

Variables
Z       Zielfunktionswert
;

Equations
1_ObjFunc          Minimierung der Gesamtdurchlaufzeit

2_Einmal(j, s)      Jeden Arbeitsgang einmal abschliessen
3_Schritte(j, s)    Einhaltung der Arbeitsplaene aller Jobs
4_Zeitpunkte(j, s)  Kopplung stetiger und binaerer Variablen
5_Ressourcen(r, t)  Kapazitätsgrenzen der Ressourcen r zum Zeitpunkt t
;

* Modell
1_ObjFunc..
    Z =e= c * sum((j, s)\$(ord(s)=ls(j)), TF(j, s));


2_Einmal(j, s)\$SJ(s, j)..
    \textbf{sum}(t, x(j, s, t)) =e= 1;

3_Schritte(j, s)\$(ord(s)>=2 and ord(s)<=ls(j))..
    TF(j, s) =g= TF(j, s-1) + d(j, s);

4_Zeitpunkte(j, s)$SJ(s, j)..
    sum(t, ord(t) * x(j, s, t)) =e= TF(j, s);

5_Ressourcen(r, t)..
    sum(j,
     sum(s$SJ(s, j),
      sum(tau$((ord(tau)>=ord(t)) and (ord(tau)<=ord(t)+d(j,s)-1)),
       a(j, s, r) * x(j, s, tau)))) =l= b(r, t);