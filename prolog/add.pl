digit(X) :- between(0, 9, X).
logic_forward([X,Y],Z) :- digit(X), digit(Y), Z is X + Y.