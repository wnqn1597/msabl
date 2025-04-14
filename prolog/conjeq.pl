digit(0).
digit(1).

logic_forward([X,Y,Z], _) :- digit(X), digit(Y), digit(Z), Z is X /\ Y.