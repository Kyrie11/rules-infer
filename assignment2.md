Question 1:
1. A game is monotonic if the worth of a coalition never decreases when a player joins a larger coalition.
we have $v(\{a\})=2 \leq min\{v(\{a,b\})=10, v(\{a,c\})=8, v(\{a,b,c\})=16\}$
similarly, $v(\{b\})=5 \leq min\{10,13,16\}$, $v(\{c\})=6 \leq min\{8,13,16\}$,
$v(\{a,b\})=10 < 16$, $v(\{b,c\})=13 < 16$, $v(\{a,c\})=8 < 16$, consequently, this game is monotonic

Besides, we need to traverse $v(S \cup T)\geq v(S)+v(T)$

$\{a\}, \{b\} : 10 \geq 2+5=7$ <br>
$\{a\}, \{c\} : 8 \geq 2+6=8$<br>
$\{b\}, \{c\} : 13 \geq 5+6=11$<br>
$\{a\}, \{b,c\} : 16 \geq 2+13=15$<br>
$\{b\}, \{a,c\} : 16 \geq 5+8=13$<br>
$\{c\}, \{a,b\} : 16 \geq 6+10=16$<br>
thus, this game is superadditive <br>
2. we have <br>
| order      | Δa | Δb | Δc | <br>
| a, b, c    | 2  | 8  | 6  |<br>
| a, c, b    | 2  | 8  | 6  |<br>
| a, b, c    | 5  | 5  | 6  |<br>
| b, c, a    | 3  | 5  | 8  |<br>
| c, a, b    | 2  | 8  | 6  |<br>
| c, b, a    | 3  | 7  | 6  |<br>
for example, when the order is $\{a,b,c\}$, we have $V(a)=2$, $V(b)=V(\{a,b\})-V(a)=8$, $V(c)=V(\{a,b,c\})-V(\{a,b\})=6$
,the other situation follows the same step<br>
then, $\varphi_a = (2+2+5+3+2+3)/6=\frac{17}{6}$, $\varphi_b=\frac{41}{6}$, $\varphi_c=\frac{38}{6}$

3. we define the effectiveness of a,b,c is $x_a, x_b, x_c$ correspondingly, then, we have:<br>
$x_a+x_b+x_c=16(1)$,<br>
$x_a\geq v(\{a\})=2, x_b\geq 5, x_c\geq 6 (2)$, <br>
$x_a+x_b\geq 10, x_a+x_c\geq 8, x_b+x_c\geq 13(3)$
combine $x_a+x_b\geq 10$, $x_c\leq 6$, $x_c\geq 6$<br>,
we have $x_c=6,x_a+x_b=10$ <br>
, since $x_b+x_c\geq 13$, we can get $x_b\geq 7$, besides, we have $x_a\geq 2$, we can get $2 \leq x_a\leq3$,<br>
$x_b=10-x_a \in [7,8]$, consequently, the core is a line segment:
$C(v)=\{(x_a,x_b,x_c)|x_c=6, x_a\in [2,3],x_b=10-x_a\}$, the endpoints are (2,8,6) and (3,7,6)

Question 2: <br>
1. there exists an NS-deviation player, $b$ wants to deviate from $\{a,b\}$ to $\{b,c\}$ since $\{b,c\} \succ_b \{a,b\}$
2. there is no IS-deviation in this coalition, since there is only one NS-deviation player $b$, however, $\{c\} \succ_c \{b,c\}$, that means $c$ doesn't accept this deviation


3. $
\begin{array}{|c|c|c|c|}
\hline
 partition& NS-deviation & player\rightarrow deviation& preference \\ \hline
 \{\{a,b,c\}\} & yes & c \rightarrow \emptyset& \{c\} \succ_c \{a,b,c\}   \\ \hline
 \{\{a,b\},\{c\}\}& yes & b \rightarrow \{c\} &  \{b,c\} \succ_b \{a,b\} \\ \hline
 \{\{a,c\},\{b\}\}& yes & a \rightarrow \{b\} &  \{a,b\} \succ_a \{a,c\}  \\ \hline
 \{\{b,c\},\{a\}\}& yes & c \rightarrow \{a\} &  \{a,c\} \succ_c \{b,c\}  \\ \hline
 \{\{a\},\{b\}\,\{c\}\}& yes & a \rightarrow \{b\} &  \{a,b\} \succ_a \{a\}  \\ \hline 
\end{array}
$
we can get the conclusion that there is not NS-stable partition

Question 3
1. Due to definition 3, we have $\sum_{j\neq i}\geq v(N \backslash \{i\})$, since $\sum_{i\in N}a_i=v(N)$, 
we can get $a_i = v(N)-\sum_{j\neq i}a_j \leq v(N)-v(N\backslash \{i\})=M_i(N,v)$
2. case A: $C \not\subseteq S$, due to the veto nature, we can get $v(S)=0$, besides, with the combination of imputation nature and $v(\{i\}) \geq 0$, 
we can get $a_i\geq 0$, then $\sum_{i\in S}a_i \geq 0=v(S)$. case B: $C \subseteq S$, then $N\backslash S \subseteq N\backslash [assignment2.md](assignment2.md)C$, due to the suppose $a_i\leq M_i(N,v) $ for all $i\in N\backslash C$,
we can get to $\sum_{i\in S}a_i =v(N)-\sum_{j\in N\backslash S}a_j \geq v(N)-\sum_{j\in N\backslash S}M_j(N,v)$, 
due to the Union property, we can get to $\sum_{i\in S}\geq v(N)-\sum_{j\in N\in S}M_j(N,v)\geq v(S)$
thus, in each case, we can have $\sum_{i\in S}a_i \geq v(S)$, due to the definition 4 and previous suppose, we can have the core conclusion.
3. since the convex nature: for any $S, T \subseteq N, v(S\cup T)+v(S\cap T)\geq v(S)+v(T)$, case A: $C\not\subseteq S$ and $C \not\subseteq T$:
if $C\not\subseteq S\cup T$,then $v(S)=v(T)=v(S\cup T)=v(S\cap T)=0$, the convex nature exists; if $C\subseteq S\cup T$ but $C\not\subseteq S$ and $C\not\subseteq T$, then $v(S)=v(T)=v(S\cap T)=0$,
however, $v(S\cup T)\geq 0$, the convex nature exists; if $C\subseteq T$ and $C\not\subseteq S$, then $v(S)=v(S\cap T)=0$, we need to derive that $v(S\cup T)\geq v(T)$, we can derive it by:
$v(X)=v(N)-\sum_{i\in N\backslash X}M_i(N,v) (C\subseteq X)$, besides, since $N\backslash (S\cup T) \subseteq N\backslash T$ and $M_i(N,v)\geq 0$, then:
$v(S\cup T)=v(N)-\sum_{i\in N\backslash (S\cup T)}M_i\geq v(N)-\sum_{i\in N\backslash T}M_i=v(T)$

case B: $C\subseteq S$ and $C\subseteq T$, then each subset contains $C$, they all can derive:$v(X)=v(N)-\sum_{i\in N\backslash X}M_i(N,v)$,
then: $\sum_{i\in N\backslash S}M_i+\sum_{i\in N\backslash T}M_i \geq \sum_{i\in N\backslash (S\cup T)}M_i+\sum_{i\in N\backslash (S\cap T)}M_i$, with $(N\backslash S)\cup (N\backslash T)=N\backslash (S\cap T), N(\backslash S)\cap (N\backslash T)=N\backslash (S\cup T)$,
we can get to :$v(S\cup T)+ v(S\cap T)=v(S)+v(T)$