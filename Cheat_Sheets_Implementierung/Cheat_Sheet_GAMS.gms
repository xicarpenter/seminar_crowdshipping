$title Ablaufplanung

* Hinweis: F�r die im Modell angegebene Instanz lassen
* sich in der GAMS-Demo-Version mit maximal 50 ganzzahligen
* oder binaeren Variablen nur die beiden ersten Modelle
* korrekt loesen, ohne ein Lizenz-Problem auszuloesen.
* Dazu ist die Anzahl der Perioden auf maximal 7 zu
* setzen. Setzt man diese Zahl auf einen groesseren
* Wert, beispielsweise auf 20 Perioden, so lassen
* sich alle Modelle korrekt loesen, dies erfordert
* aber die GAMS-Vollversion.

Sets
j        Jobs
r        Ressourcen
s        Schritte
t        Perioden

SJ(s, j)    Schritte des Jobs j
;

alias(t,tau);

Parameter
a(j,s,r) benötigte Kapazitätseinheiten der Ressource r für Schritt s des Jobs j
b(r,t)   Kapazität der Ressource r in Periode t
c        Umrechnungsfaktor von Perioden in Stunden
d(j,s)   Dauer von Schritt s des Jobs j
ls(j)    letzter Arbeitsschritt von Job j (oder: Anzahl der Schritte von Job j)
;

Positive Variables
TF(j,s)  Fertigstellungszeitpunkt von Schritt s des Jobs j
;

Binary Variables
x(j,s,t) binäre Variable mit Wert 1 wenn in Periode t der Schritt s des Jobs j beendet wird 0 sonst
;

Variables
Z       Zielfunktionswert
;

Equations
ObjFunc          Minimierung der Gesamtdurchlaufzeit

Einmal(j,s)      Jeden Arbeitsgang einmal abschliessen
Schritte(j,s)    Einhaltung der Arbeitsplaene aller Jobs
Zeitpunkte(j,s)  Kopplung stetiger und binaerer Variablen
Ressourcen(r,t)  Kapazitaetsgrenzen der Ressourcen r zum Zeitpunkt t
;

ObjFunc..
Z =e= c * sum((j,s)$(ord(s)=ls(j)), TF(j, s))
;


Einmal(j,s)$SJ(s, j)..
         sum(t, x(j,s,t))=e=1
;

Schritte(j,s)$(ord(s)>=2 and ord(s)<=ls(j))..
         TF(j,s) =g= TF(j,s-1)+d(j,s)
;


Zeitpunkte(j,s)$SJ(s, j)..
         sum(t, ord(t)*x(j,s,t)) =e= TF(j,s)
;

Ressourcen(r,t)..
         sum(j,
            sum(s$SJ(s, j),
                sum(tau$((ord(tau)>=ord(t)) and (ord(tau)<=ord(t)+d(j,s)-1)),
                      a(j,s,r) * x(j,s,tau)))) =l= b(r,t)
;




* Daten der konkreten Instanz

sets     j /j1*j4/
         r /rA,rB,rC/
         s /s1*s2/
         t /t1*t20/;

parameter
ls(j)    /j1 2, j2 2, j3 2, j4 1/;


* Dauern der Prozessschritte
table d(j,s)
                 s1      s2
         j1      3       2
         j2      1       3
         j3      3       2
         j4      4       0      ;


SJ(s, j)=no;

loop(j,
    loop(s,
        if(ls(j) >= ord(s),
                SJ(s, j) = yes;
        );
    );
);

Sets
JSR(j, s, r);

* Jeder Schritt erfordert stets eine Ressourceneinheit

a(j,s,r)=0;
a('j1','s1','rA')=1;
a('j1','s2','rB')=1;

a('j2','s1','rA')=1;
a('j2','s2','rC')=1;

a('j3','s1','rB')=1;
a('j3','s2','rA')=1;

a('j4','s1','rC')=1;


* Periodenkapazitaet konstant
b(r,t)=1;

*Umrechnungsfactor konstant
c = 1;


Model Ablaufplanung /ObjFunc, Einmal, Schritte,
                      Zeitpunkte, Ressourcen/;

Ablaufplanung.limrow=1000;
Ablaufplanung.limcol=1000;
Ablaufplanung.optcr=0.0;




solve Ablaufplanung minimizing Z using mip;


