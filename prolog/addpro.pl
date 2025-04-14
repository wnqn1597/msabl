:- use_module(library(lists)).
:- use_module(library(clpfd)).

%% Digit
digit(X) :- between(0, 9, X).

digits([]).
digits([D|T]) :- digit(D), digits(T).

%% Get number
get_number([], 0).
get_number([D|T], V) :-
    get_number(T, U), V is D + U * 10.

number(L, V) :-
    digits(L), reverse(L, R), get_number(R, V).

%% valid num (4 digits)
valid(N) :- between(0, 10000, N).

%% add
logic_forward([L1,L2], A) :-
    valid(N1), N2 is A - N1, valid(N2), number(L1, N1), number(L2, N2).