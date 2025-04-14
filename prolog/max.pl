digit(N) :- between(0, 3, N).
logic_forward([], 0).
logic_forward([E|T], Res) :- digit(E), logic_forward(T, Res2), Res is max(E, Res2).
