---
layout: post
mathjax: true
comments: true
title:  "Formal Methods"
---
This blog post is about formal methods: the logic and algorithms behind mathematically proving that a system meets its specification. It is organized into parts. The first follows the book *Model Checking* — exhaustively checking a system model against a temporal-logic specification. The second follows the book *Decision Procedures* — the SAT and SMT engines that decide the logical formulas such checks (and program verification in general) ultimately reduce to. The third, *Automated Reasoning*, steps up to first-order logic, theorem proving, and the deductive program-logic frameworks (Hoare logic, VC generation, abstract interpretation) that sit on top of those engines.

* TOC
{:toc}

# Model Checking

[Model Checking, 2nd Edition](https://mitpress.mit.edu/9780262038836/model-checking/) by Edmund Clarke, Orna Grumberg, Daniel Kroening, Doron Peled, and Helmut Veith covers the foundations of automatic verification for finite-state systems. The key idea: given a system model and a specification in temporal logic, an algorithm exhaustively checks whether the model satisfies the specification, producing a counterexample if it doesn't. This part summarizes the main ideas from chapters 1-6, 8-11, and 13-14.

## Modeling Systems

The first step is converting a system into a **Kripke structure** $M = (S, S_0, R, AP, L)$:
- $S$: finite set of states
- $S_0 \subseteq S$: initial states
- $R \subseteq S \times S$: transition relation (must be left-total: every state has a successor)
- $AP$: atomic propositions
- $L: S \to 2^{AP}$: labels each state with propositions true in it

States are valuations of system variables. We use **first-order logic** to represent sets of states and transitions symbolically:
- $\mathcal{S}_0(s)$: formula for initial states
- $\mathcal{R}(V, V')$: formula for transitions, using current-state variables $V$ and next-state variables $V'$

### Modeling programs

A program counter $pc$ tracks control flow. Each statement type (assignment, conditional, while) translates into a disjunct of $\mathcal{R}$. The full transition relation is a **disjunction** of all statement formulas.

For **concurrent programs** with interleaving semantics, exactly one process moves at a time. The transition relation is a disjunction over processes, where each process's transition is conjoined with a frame condition ($\text{same}(V \setminus V_i)$) ensuring other variables don't change.

For **synchronous circuits**, the transition relation is a **conjunction** of individual register update functions. For **asynchronous circuits**, it is a **disjunction** (interleaving).

### Fairness

A **fair Kripke structure** adds fairness constraints $F = \{P_1, \ldots, P_k\}$ where each $P_i \subseteq S$. A path is fair if it visits each $P_i$ infinitely often. This prevents unrealistic behaviors like one process starving forever.

## Temporal Logic

Temporal logic specifies dynamic behavior over computation trees (unwindings of the Kripke structure from a state).

### CTL\*

CTL\* has **path quantifiers** (A = all paths, E = exists a path) and **temporal operators** (X = next, F = eventually, G = globally, U = until, R = release). Formulas are either **state formulas** (true at a state) or **path formulas** (true along a path).

### Important sublogics

**CTL** (Computation Tree Logic): path quantifiers and temporal operators always occur in pairs: $AX, EX, AF, EF, AG, EG, AU, EU, AR, ER$. Every subformula is a state formula. All operators reduce to $EX$, $EG$, and $EU$.

**LTL** (Linear Temporal Logic): formulas have the form $Af$ where $f$ is a path formula without path quantifiers. A single universal quantifier over all paths.

**CTL and LTL are incomparable in expressiveness.** CTL\* subsumes both.

**Example: $A(FGp)$ is LTL but not CTL.** This says "on every path, eventually $p$ holds forever *on that path*." The closest CTL approximation is $AFAGp$: "on all paths, eventually reach a state where $p$ holds on ALL future paths." But these are different — $A(FGp)$ forgives unused bad branches, $AFAGp$ does not:

```
     ┌──┐
     ▼  │
──► (s) ┘───► (t) ───► (u)
    [p]       [¬p]     [p] ◄─┐
                         └────┘
```

$A(FGp)$: **YES**. Path s→s→s→...: $p$ forever ✓. Path s→t→u→u→...: $p$ from $u$ onward ✓. Each path individually settles into $p$.

$AFAGp$: **NO**. Is $AGp$ true at $s$? No — $s$ can reach $t(\neg p)$. The path s→s→s→... never leaves $s$ where $AGp$ is always false, so $AFAGp$ fails.

**Example: $AG(EFp)$ is CTL but not LTL.** This says "from every reachable state, there exists some path that eventually reaches $p$." LTL can only talk about individual paths — it cannot express "there exists a path" from an intermediate state. No LTL formula can distinguish a structure where every state has *some* path to $p$ from one where every state has *all* paths reaching $p$.

**Example: $A(FGp) \lor AG(EFp)$ is CTL\* but neither CTL nor LTL.**

### Safety vs liveness

- **Safety** ($AGp$): something bad never happens. Counterexample is a finite path to a bad state.
- **Liveness** ($AFp$, $A(pUq)$): something good eventually happens. Counterexample is a **lasso** (finite stem + repeating loop).

### Complexity

| Logic (compact description) | Space | Time (naive) |
|---|---|---|
| CTL | PSPACE-complete | $O(2^n \cdot \|f\|)$ |
| LTL | PSPACE-complete | $O(2^n \cdot 2^{\|f\|})$ |
| CTL\* | PSPACE-complete | $O(2^n \cdot 2^{\|f\|})$ |

With explicit state graph $\|M\|$: CTL is $O(\|f\| \cdot (\|S\| + \|R\|))$, linear in both formula and model. LTL is linear in the model but exponential in the formula.

## CTL Model Checking

### Explicit-state algorithm

Process subformulas bottom-up. Label each state with the subformulas it satisfies. Since every CTL formula reduces to $\neg, \lor, EX, EU, EG$, handle six cases:

- **$EXf_1$**: label states with a successor labeled $f_1$
- **$E(f_1 U f_2)$**: find states labeled $f_2$, then work backward through states labeled $f_1$
- **$EGf_1$**: restrict to states satisfying $f_1$, find nontrivial **strongly connected components** (SCCs), then work backward. A state satisfies $EGf_1$ iff it can reach a nontrivial SCC where all states satisfy $f_1$.

Complexity: $O(\|f\| \cdot (\|S\| + \|R\|))$.

**Example (microwave oven, Figure 5.3 in the book).** The microwave has 7 states labeled with Start, Close, Heat, Error. Check $AG(\text{Start} \to AF\text{Heat})$. Rewrite as $\neg EF(\text{Start} \land EG\neg\text{Heat})$. Work inside-out:
1. $\neg\text{Heat}$ states: $\{1,2,3,5,6\}$. Within these, the nontrivial SCC is $\{1,2,3,5\}$ (cycle: 1→2→5→3→1). State 6 only goes to 7 (Heat), so it can't stay in $\neg$Heat. Thus $[\![EG\neg\text{Heat}]\!] = \{1,2,3,5\}$.
2. $[\![\text{Start} \land EG\neg\text{Heat}]\!] = \{2, 5\}$ (states with Start in the SCC).
3. $[\![EF(\text{Start} \land EG\neg\text{Heat})]\!]$: work backward from $\{2,5\}$ — all states can reach 2 or 5, so this is $\{1,2,3,4,5,6,7\}$.
4. Negate: $[\![AG(\text{Start} \to AF\text{Heat})]\!] = \emptyset$.
5. Initial state 1 is not in $\emptyset$, so the formula does **not** hold — the oven can loop in $\{1,2,3,5\}$ without ever heating.

### Fixpoint characterization

CTL operators have elegant fixpoint characterizations:

$$E(f_1 U f_2) = \mu Z.\; f_2 \lor (f_1 \land EXZ)$$

$$EG\, f_1 = \nu Z.\; f_1 \land EXZ$$

Least fixpoint ($\mu$) = start from $\emptyset$, apply $\tau$ until convergence (eventualities). Greatest fixpoint ($\nu$) = start from $S$, shrink until convergence (invariants). Convergence is guaranteed in at most $\|S\|$ iterations since $S$ is finite.

### Fairness

The explicit-state algorithm for $EGf_1$ uses nontrivial SCCs. For fair model checking with constraints $F = \{P_1, \ldots, P_n\}$, the only change is requiring **fair** MSCCs — those intersecting every $P_i$. Complexity becomes $O(\|f\| \cdot (\|S\| + \|R\|) \cdot \|F\|)$.

The key operator is $E_f G f_1$: "there exists a fair path where $f_1$ holds globally." Its fixpoint characterization is:

$$E_f G f_1 = \nu Z.\; f_1 \land \bigwedge_{k=1}^{n} EX\, E(f_1 \, U \, (Z \land P_k))$$

This says: $Z$ is the largest set where every state (1) satisfies $f_1$, and (2) for each fairness constraint $P_k$, can reach a state in $Z \cap P_k$ along a path where $f_1$ holds throughout. The outer greatest fixpoint ensures the path can be extended forever; the inner $EU$ ensures each $P_k$ is visited. This formula nests both fixpoints — the $EU$ inside is a least fixpoint inside the greatest fixpoint of $Z$.

Once we have $E_f G$, we derive a predicate $\text{fair} = E_f G \text{true}$ — the set of states from which some fair path exists. The other operators reduce to unfair versions by conjoining with $\text{fair}$:

- $E_f X f_1 \equiv EX(f_1 \wedge \text{fair})$ — the successor must be on a fair path
- $E_f(f_1 U f_2) \equiv E(f_1 U (f_2 \wedge \text{fair}))$ — the $f_2$-state must be on a fair path

This works because the only place unfair paths cause trouble is in allowing infinite looping without visiting every $P_k$. By requiring the "witness" state to be in $\text{fair}$, we ensure the path can be extended fairly from that point.

**Example (microwave with fairness).** We saw that $AG(\text{Start} \to AF\text{Heat})$ fails without fairness because of the SCC $\{1,2,3,5\}$ looping without heating. But this SCC represents the user repeatedly starting the oven with the door open — unrealistic behavior. Add the fairness constraint $F = \{P\}$ where $P = \{s \mid s \models \text{Start} \land \text{Close} \land \neg\text{Error}\}$ = $\{6, 7\}$ (the user operates correctly infinitely often — starting with door closed and no error).

Now recheck: the SCC $\{1,2,3,5\}$ within $\neg$Heat states does **not** intersect $P = \{6,7\}$, so it is not a fair SCC. There are no fair nontrivial SCCs within $\neg$Heat states at all. Thus $E_f G\neg\text{Heat} = \emptyset$, the negated formula $EF(\text{Start} \land E_fG\neg\text{Heat})$ is also empty, and $AG(\text{Start} \to AF\text{Heat})$ **holds** under fairness.

The fairness constraint eliminated the unrealistic looping path. Any fair path must visit states 6 or 7 infinitely often, which forces the oven to eventually go through the normal cycle: close door → start oven (state 6) → warmup → heat (state 7).

## LTL Model Checking via Tableau

To check $Af$ on $M$: check $E\neg f$ and negate.

### Tableau construction

Given path formula $f$, build a **tableau** $T$:
- States are subsets of **elementary formulas** $el(f)$ (atomic propositions and $Xg$ subformulas): $S_T = \mathcal{P}(el(f))$
- Initial states: $sat(f)$
- Transitions: $s \to s'$ iff for every $Xg \in el(f)$: $s \in sat(Xg) \Leftrightarrow s' \in sat(g)$
- Fairness: for each $gUh$ in $f$, require infinitely often visiting $sat(\neg(gUh) \lor h)$

The fairness constraint prevents paths from eternally promising $gUh$ without ever delivering $h$. The disjunct $\neg(gUh)$ handles the case where the until obligation is no longer active — requiring $h$ infinitely often would be too strong, since $gUh$ only needs $h$ to be delivered once.

### Tableau example

For the formula $g = (\neg\text{heat}) U \text{close}$, the elementary formulas are $el(g) = \{\text{heat}, \text{close}, Xg\}$. The tableau has $2^3 = 8$ states, each a subset of $el(g)$:

| State | Contents | In $sat(g)$? |
|---|---|---|
| 1 | $\{\neg h, c, Xg\}$ | yes (has $c$) |
| 2 | $\{h, c, Xg\}$ | yes (has $c$) |
| 3 | $\{\neg h, \neg c, Xg\}$ | yes ($\neg h$ and $Xg$) |
| 4 | $\{h, c\}$ | yes (has $c$) |
| 5 | $\{h, \neg c, Xg\}$ | no |
| 6 | $\{\neg h, c\}$ | yes (has $c$) |
| 7 | $\{\neg h, \neg c\}$ | no |
| 8 | $\{h, \neg c\}$ | no |

States with $Xg$ (states 1,2,3,5) transition to states in $sat(g)$. States without $Xg$ (states 4,6,7,8) transition to states in $sat(\neg g)$. Transition is biconditional: $s \in sat(Xg) \Leftrightarrow s' \in sat(g)$.

The fairness constraint is $F_T = sat(\neg g \lor \text{close}) = \{5,7,8\} \cup \{1,2,4,6\} = S_T \setminus \{3\}$. This prevents looping forever in state 3 (which is in $sat(g)$ but never satisfies $\text{close}$, so the until is never discharged).

### Product and CTL reduction

Compute product $P = T \times M$ (states agreeing on atomic propositions). Then: $M, s' \models Ef$ iff some $(s, s') \in sat(f)$ starts a **fair path** in $P$.

Finding states with fair paths is exactly **$EG\text{true}$ with fairness** — a CTL problem! So LTL model checking reduces to fair CTL model checking on the product.

For $Af$: compute states satisfying $E\neg f$, take complement, check if it contains all initial states.

### CTL\* model checking

Combine CTL and LTL algorithms. Process subformulas by **level**: atomic propositions at level 0, then formulas whose state subformulas are all at lower levels. At each level, if the formula is CTL, use CTL model checking; otherwise treat it as LTL. Complexity: same as LTL, $O(\|M\| \cdot 2^{\|f\|})$.

## Symbolic Model Checking with BDDs

### Ordered Binary Decision Diagrams

**OBDDs** are canonical representations of Boolean functions as directed acyclic graphs with a fixed variable ordering. Key properties:
- Canonical: two functions are equivalent iff their OBDDs are isomorphic
- Often exponentially smaller than truth tables
- Efficient operations: Apply (any binary op) runs in $O(\|f\| \cdot \|g\|)$

Variable ordering critically affects size. Finding optimal ordering is NP-complete, but heuristics (depth-first circuit traversal, dynamic reordering) work well in practice.

### Symbolic CTL model checking

Represent states, transitions, and sets as OBDDs. The fixpoint-based CTL algorithm translates directly:
- $EXf$: compute $\exists V'.\; f(V') \land R(V, V')$ using **relational product**
- $E(f U g)$: iterate $Q := g \lor (f \land EXQ)$ from $Q = \emptyset$
- $EGf$: iterate $Q := f \land EXQ$ from $Q = S$

Compare OBDDs for convergence (canonical form makes equality check trivial).

### Does symbolic model checking handle LTL?

Yes, through the same tableau reduction. To check LTL formula $Af$ on $M$:
1. Build the tableau $T_{\neg f}$ for $\neg f$ (its states are Boolean — subsets of $el(f)$ — so it has an OBDD representation)
2. Compute the symbolic product $P = T_{\neg f} \times M$ (conjunction of their transition relations, restricted to states agreeing on atomic propositions)
3. Run symbolic fair CTL model checking for $EG\text{true}$ on $P$ with fairness constraints $F_T$
4. Intersect the result with $sat(\neg f)$, project onto $M$'s states, check against $S_0$

The entire pipeline stays symbolic — no explicit state enumeration. The exponential blowup from LTL ($2^{|f|}$ tableau states) manifests in the OBDD size rather than explicit enumeration, and BDDs often handle this compactly due to structural regularity.

### Partitioned transition relations

For synchronous circuits, $R = \bigwedge_i R_i$ (**conjunctive partitioning**). For asynchronous, $R = \bigvee_i R_i$ (**disjunctive partitioning**). Key optimization: **early quantification** — eliminate variables as soon as no remaining $R_i$ depends on them, avoiding building the monolithic transition relation.

## SAT-Based Model Checking

When the model is too large for BDDs, we can instead pose model-checking questions as propositional satisfiability problems and hand them to a **SAT solver** (see the [SAT](#sat) section for the algorithm itself).

### DPLL and CDCL

SAT solvers determine satisfiability of propositional CNF formulas. Modern solvers use:
- **DPLL**: binary search with backtracking
- **BCP** (Boolean Constraint Propagation): when a clause becomes unit (one unassigned literal), force that literal
- **CDCL** (Conflict-Driven Clause Learning): when a conflict occurs, analyze the **implication graph**, derive a **conflict clause** via resolution, add it to prevent repeating the same conflict. Enables **non-chronological backtracking**.

### Bounded Model Checking (BMC)

Unwind the model $k$ steps. For $AGp$:

$$\text{path}_k(s_0, \ldots, s_k) \land \bigvee_{i=0}^{k} \neg p(s_i)$$

Satisfiable iff a counterexample of length $\leq k$ exists. For $AFp$ (liveness), require a **lasso**: the path must loop back, with $\neg p$ on every state.

Increase $k$ until either a counterexample is found or $k$ exceeds a **completeness threshold** (e.g., the diameter of the state graph for $AGp$). Note: the completeness threshold depends on both the model and the property, and computing a tight one is as hard as model checking itself.

### BMC for full LTL

BMC handles arbitrary LTL formulas $A\varphi$ by reusing the **tableau construction** from Chapter 6. Build the tableau $T_{\neg\varphi}$ with fairness constraints, form the product $\Psi = M \times T_{\neg\varphi}$, then search for a **fair lasso** of length $k$ in $\Psi$:

$$\Psi_0(s_0) \land \bigwedge_{i=0}^{k-1} R_\Psi(s_i, s_{i+1}) \land \text{lasso}_k \land \bigwedge_{P \in F_\Psi} \text{fair}_P$$

where $\text{lasso}_k = \bigvee_{i=0}^{k-1}(s_k = s_i)$ requires the path to loop back, and $\text{fair}_P$ requires each fairness constraint $P$ to be visited within the loop. The formula is linear in $k$ and can be passed directly to a SAT solver. This reduces full LTL model checking to propositional satisfiability, at the cost of only finding counterexamples up to length $k$.

### k-Induction

Prove $AGp$ without reaching the completeness threshold:
- **Base case**: BMC with bound $k-1$ (no counterexample of length $\leq k-1$)
- **Step case**: any path of $k$ states satisfying $p$ has a successor satisfying $p$

Adding a **uniqueness constraint** (all states pairwise different) makes k-induction complete.

### Craig Interpolation

Given $A \land B$ unsatisfiable, Craig's theorem guarantees an **interpolant** $I$ such that $A \Rightarrow I$, $I \land B$ is unsat, and $I$ uses only variables common to $A$ and $B$.

**McMillan's interpolation system** computes $I$ from a resolution proof:
- A-clause leaf: keep only B-visible literals
- B-clause leaf: $\text{true}$
- Internal node, pivot in B (shared): **conjunction** $Itp(v^+) \land Itp(v^-)$
- Internal node, pivot not in B (A-local): **disjunction** $Itp(v^+) \lor Itp(v^-)$

Intuition: the interpolant is "A's contribution to the contradiction, expressed in shared language." Conjunction for shared pivots because both claims are unconditional facts from A. Disjunction for A-local pivots because B can't see the private variable, so A can only promise one of two possibilities.

### CraigReachability

Uses interpolation to find an **inductive invariant** for $AGp$:
1. Start with $Q = S_0$, build $A = Q(s_0) \land R(s_0, s_1)$ and $B = \bigwedge R(s_i, s_{i+1}) \land \bigvee \neg p(s_i)$
2. If $A \land B$ is UNSAT: interpolant $I$ overapproximates $\text{post-image}(Q)$ and excludes states reaching $\neg p$ in $k-1$ steps
3. If $I \subseteq Q$: $Q$ is an inductive invariant, $AGp$ holds
4. Otherwise: $Q := Q \cup I$, repeat

If $A \land B$ is SAT with $Q = S_0$: genuine counterexample. If SAT with $Q \neq S_0$: spurious, reset $Q := S_0$, increase $k$ (larger $k$ produces tighter interpolants).

### Property-Directed Reachability (PDR/IC3)

Uses **frames** $F_0 \subseteq F_1 \subseteq \ldots \subseteq F_k$, each overapproximating reachable states at depth $i$. Invariants:
1. $S_0 \subseteq F_0$
2. $F_i \subseteq F_{i+1}$ (monotone)
3. No $\neg p$ states in any $F_i$
4. $\text{post-image}(F_i) \subseteq F_{i+1}$

When $F_i = F_{i+1}$ for some $i$: $F_i$ is an inductive invariant, done. Key advantage over interpolation: uses only **one copy** of the transition relation (no unwinding), so more memory efficient.

## Equivalences and Preorders

### Bisimulation

A relation $B \subseteq S \times S'$ is a **bisimulation** if whenever $B(s, s')$:
1. Same labeling: $L(s) = L'(s')$
2. Every successor of $s$ has a corresponding successor of $s'$ (and vice versa)

Bisimulation equivalent structures satisfy exactly the same **CTL\*** formulas. Conversely, if two structures satisfy the same CTL formulas, they are bisimulation equivalent. (This does not mean CTL = CTL\* in expressiveness — it means CTL is expressive enough to *distinguish* any two non-bisimilar structures, even though it cannot express all CTL\* properties.)

**Example.** Consider two structures: $M$ has state $a$ with two successors labeled $c$ and $d$. $M'$ has state $a$ with one successor labeled $b$, and $b$ has two successors $c$ and $d$. These are bisimilar (the extra intermediate step in $M'$ just unwinds $M$), so they satisfy the same CTL\* formulas.

Now consider $M$ where state $b$ has successors $c$ and $d$ separately, vs $M'$ where a single state $b$ has both a $c$-successor and a $d$-successor. If the branching structure differs (e.g., in $M$ two different $b$-states each lead to only one of $c,d$), they are not bisimilar and can be distinguished by CTL formula $AG(b \to EXc)$.

### Simulation

A **simulation** $H$ from $M$ to $M'$ requires only the forward direction: every successor of $s$ has a matching successor of $s'$, but not vice versa. $M'$ **simulates** $M$ means $M'$ has all behaviors of $M$ (and possibly more).

If $M \preceq M'$ (simulation preorder), then for every **ACTL\*** formula $f$: $M' \models f \Rightarrow M \models f$. Dually, for ECTL\*: $M \models f \Rightarrow M' \models f$. This is the theoretical foundation for abstraction.

**Why ACTL\* and not full CTL\*?** Simulation means $M'$ has all behaviors of $M$ plus possibly more. Universal properties ($A$) are preserved because adding behaviors can only make them harder to satisfy — if $M'$ satisfies $A\varphi$ despite having more behaviors, $M$ does too. But existential properties ($E\varphi$) go the other direction: $M \models E\varphi$ implies $M' \models E\varphi$ since $M$'s witness path exists in $M'$ too. Mixing $A$ and $E$ (full CTL\*) breaks preservation in both directions.

**Example.** Two structures where each simulates the other (mutual simulation) but they are not bisimilar: they satisfy the same ACTL\* formulas but can be distinguished by the CTL formula $AG(b \to EXc)$, which uses the existential quantifier $E$ inside the universal $AG$.

## Abstraction

### Existential abstraction

Given an abstraction function $\alpha: S \to \hat{S}$ mapping concrete states to abstract states:
- Abstract initial states: $\hat{s} \in \hat{S}_0$ if some $s \in S_0$ has $\alpha(s) = \hat{s}$
- Abstract transitions: $\hat{R}(\hat{s}, \hat{t})$ if some $R(s, t)$ with $\alpha(s) = \hat{s}$, $\alpha(t) = \hat{t}$

This guarantees $M \preceq \hat{M}$ (simulation), so ACTL\*/LTL properties proved on $\hat{M}$ hold on $M$.

### Three common abstractions

**Localization reduction** (hardware): partition variables into visible/invisible. Invisible variables become unconstrained inputs. Cone-of-influence reduction is a conservative choice.

**Data abstraction**: abstract each variable's domain separately. E.g., integers to $\{a^-, a^0, a^+\}$.

**Predicate abstraction** (software): choose predicates $P_1, \ldots, P_k$ over program variables. Abstract states are Boolean valuations of these predicates. The concrete program becomes a **Boolean program**.

### CEGAR

Counterexample-Guided Abstraction Refinement:
1. **Abstract**: generate initial (coarse) abstraction
2. **Model check**: if property holds on $\hat{M}$, done (property holds on $M$)
3. **Check counterexample**: simulate abstract counterexample on $M$. If feasible, done (real bug). Otherwise it's **spurious**.
4. **Refine**: identify the **failure state** (where the concrete path dies), separate **dead-end states** from **bad states** by splitting the abstract state. Go to step 2.

CEGAR is guaranteed to terminate for finite-state systems: each refinement strictly refines the partition, and the finest partition is isomorphic to $M$.

**Example (traffic light).** A US traffic light cycles red $\to$ green $\to$ yellow $\to$ red. We want to prove $AGAF(\text{state}=\text{red})$ ("always eventually red"). Abstract by collapsing green and yellow into $\hat{s}_{\neg\text{red}}$. The abstract model has a self-loop on $\hat{s}_{\neg\text{red}}$, producing a spurious counterexample that stays in $\hat{s}_{\neg\text{red}}$ forever. CEGAR detects this is spurious (the concrete green $\to$ yellow $\to$ red path doesn't loop), and refines by splitting $\hat{s}_{\neg\text{red}}$ into green and yellow, eliminating the spurious loop.

## Software Model Checking

### Symbolic execution

Given a program path, compute the **strongest postcondition** $sp(P, X)$:
- Assignment: $sp(v := e, X) = \exists w.\; X[v/w] \land v = e[v/w]$
- Condition: $sp(c, X) = X \land c$
- Sequence: $sp(P_1; P_2, X) = sp(P_2, sp(P_1, X))$

Check if the postcondition at an assertion is satisfiable with the assertion's negation. If yes, the assertion can be violated (feasible error path).

### Predicate abstraction for programs

Transform a program into a **Boolean program** using predicates:
1. Replace assignments with their effect on predicates (check validity of implications using a decision procedure)
2. Replace conditionals with their abstract versions
3. Unknown effects become nondeterministic ($*$)

The Boolean program overapproximates the original. Verify using BDDs or SAT. If a spurious counterexample is found, refine:
- **Transformer refinement**: remove spurious abstract transitions
- **Domain refinement**: add new predicates (e.g., from strongest postconditions along the spurious path)

# Decision Procedures

[Decision Procedures: An Algorithmic Point of View](https://link.springer.com/book/10.1007/978-3-540-74105-3) by Daniel Kroening and Ofer Strichman builds verification engines from the bottom up: a SAT solver at the core, theory solvers for richer logics (equality and uninterpreted functions, linear arithmetic, bit-vectors, arrays, pointers) layered on top, and the SMT architecture — DPLL(T) — that combines them. This part summarizes chapters 2–11, then closes with how Z3 ties everything together.

## SAT

Everything in this part bottoms out in the same question (Chapter 2): is a propositional CNF formula satisfiable? Modern solvers answer this with **CDCL** (Conflict-Driven Clause Learning) and **non-chronological backtracking**. What follows is a minimal but complete description of the algorithm — no recursion, no call stack. The entire solver is one flat loop over a **trail**: an ordered array of assigned literals that *is* the search state, where backtracking is just popping from the end.

### Data structures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CORE STATE                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  trail[]          Ordered array of assigned literals.                       │
│                   This IS the search state. Newest assignments at the end.  │
│                   Backtracking = popping from the end.                      │
│                                                                             │
│  value[lit]       Current value of a literal: TRUE, FALSE, or UNDEF.        │
│                   value[x] and value[¬x] are always complementary.          │
│                                                                             │
│  level[var]       The decision level at which this variable was assigned.   │
│                   Level 0 = forced by the original formula (always true).   │
│                   Level k>0 = assigned after the k-th decision.             │
│                                                                             │
│  reason[var]      The clause that forced this variable (its "antecedent").  │
│                   NULL if the variable was a decision (free choice).        │
│                   Used to reconstruct the implication graph during ANALYZE. │
│                                                                             │
│  dl               Integer. Current decision level.                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ CLAUSE DATABASE                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  clauses[]        All clauses: original + learned.                          │
│                   Each clause is just an array of literals.                 │
│                                                                             │
│  watch[lit]       For each literal, a list of clauses that "watch" it.      │
│                   A clause watches exactly 2 of its literals.               │
│                   Key invariant: a clause can only become unit or conflict  │
│                   when one of its watched literals is falsified.            │
│                   Backtracking NEVER needs to update watch lists.           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ HEURISTICS                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  activity[var]    VSIDS score. Bumped for variables involved in conflicts.  │
│                   All scores decayed periodically (multiply by 0.95).       │
│                   PICK-LITERAL returns the unassigned var with max score.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Main solver loop

The outer loop decides a variable, then propagates. The inner loop handles cascading conflicts: learning a clause and propagating it might immediately cause another conflict.

```
CDCL-SOLVE(F):
│
│   // Propagate any unit clauses present in the original formula.
│   // If this already conflicts, the formula is trivially UNSAT.
│   if PROPAGATE() = CONFLICT:
│       return UNSAT
│
│   // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   // MAIN LOOP — runs until all variables assigned or UNSAT proven
│   // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   while has unassigned variables:
│   │
│   │   // ─── DECIDE ───────────────────────────────────────────────
│   │   // Pick a variable and a polarity. This is a "guess."
│   │   // We record it as a decision (reason = NULL).
│   │   dl ← dl + 1
│   │   lit ← PICK-LITERAL()          // highest activity, unassigned
│   │   ASSIGN(lit, level=dl, reason=NULL)
│   │   trail.push(lit)
│   │
│   │   // ─── PROPAGATE + CONFLICT LOOP ────────────────────────────
│   │   // After each decision, propagate. If conflict, learn & jump.
│   │   // The inner while handles cascading conflicts: learning a
│   │   // clause and propagating it might cause another conflict.
│   │   while PROPAGATE() = CONFLICT:
│   │   │
│   │   │   // Conflict at level 0 means the formula itself is
│   │   │   // contradictory — no assignment can satisfy it.
│   │   │   if dl = 0:
│   │   │       return UNSAT
│   │   │
│   │   │   // Learn a clause and find where to jump back.
│   │   │   (learned, bt_level) ← ANALYZE()
│   │   │   clauses.add(learned)
│   │   │
│   │   │   // ─── BACKJUMP ─────────────────────────────────────────
│   │   │   // Pop the trail back to bt_level.
│   │   │   // This may skip many decision levels at once.
│   │   │   // After this, the learned clause is unit → PROPAGATE
│   │   │   // will immediately assign the asserting literal.
│   │   │   BACKJUMP(bt_level)
│   │   │   dl ← bt_level
│   │   │
│   │   end while
│   │
│   end while
│
│   return SAT + current assignment
```

### Unit propagation (two-watched literals)

For each literal `p` just assigned TRUE, only clauses watching `¬p` can be affected — that watched literal just became FALSE, so the clause lost a guard. The scheme is designed so backtracking never has to repair watch lists.

```
PROPAGATE():
│
│   // Process every literal that was recently assigned TRUE.
│   // For each such literal p, look at clauses watching ¬p
│   // (because ¬p just became FALSE, those clauses lost a guard).
│
│   while propagation_queue not empty:
│   │   p ← dequeue()                     // p was just set TRUE
│   │
│   │   for each clause C in watch[¬p]:    // C watches ¬p, which is now FALSE
│   │   │
│   │   │   other ← the other watched literal in C (not ¬p)
│   │   │
│   │   │   // Case 1: other watched literal is already TRUE → clause satisfied
│   │   │   if value[other] = TRUE:
│   │   │       continue                   // clause is satisfied, do nothing
│   │   │
│   │   │   // Case 2: find a replacement — any non-false literal in C
│   │   │   if exists literal w in C, w ≠ other, value[w] ≠ FALSE:
│   │   │       replace ¬p with w in C's watch pair
│   │   │       move C from watch[¬p] to watch[w]
│   │   │       continue
│   │   │
│   │   │   // Case 3: no replacement found, other is UNDEF → unit clause
│   │   │   //          force other to TRUE
│   │   │   if value[other] = UNDEF:
│   │   │       ASSIGN(other, level=dl, reason=C)
│   │   │       trail.push(other)
│   │   │       enqueue(other)             // other might trigger more propagation
│   │   │
│   │   │   // Case 4: no replacement, other is FALSE → CONFLICT
│   │   │   //          every literal in C is false
│   │   │   else:
│   │   │       conflict_clause ← C
│   │   │       return CONFLICT
│   │   │
│   │   end for
│   end while
│
│   return OK
```

### Conflict analysis (First-UIP)

Resolve the conflict clause backwards along the trail until exactly one literal at the current decision level remains — the First Unique Implication Point. That literal becomes the asserting literal of the learned clause.

```
ANALYZE():
│
│   // Goal: resolve the conflict clause backwards along the trail
│   //       until exactly ONE literal at the current decision level
│   //       remains. That literal is the First UIP.
│
│   clause ← conflict_clause
│   count ← |{ lit ∈ clause : level[var(lit)] = dl }|
│
│   i ← len(trail) - 1
│
│   // Walk the trail from newest to oldest.
│   // Each step resolves away one current-level literal.
│   while count > 1:
│   │   lit ← trail[i]
│   │   i ← i - 1
│   │
│   │   // Only resolve literals that are:
│   │   //   (a) in the current clause (negated), AND
│   │   //   (b) at the current decision level, AND
│   │   //   (c) implied (have an antecedent — not a decision)
│   │   if ¬lit ∈ clause AND level[var(lit)] = dl AND reason[var(lit)] ≠ NULL:
│   │       // Resolution: merge clause with the antecedent,
│   │       //             removing the resolved variable.
│   │       clause ← (clause ∖ {¬lit}) ∪ (reason[var(lit)] ∖ {lit})
│   │       count ← |{ l ∈ clause : level[var(l)] = dl }|
│   │
│   end while
│
│   // ─── DETERMINE BACKJUMP LEVEL ─────────────────────────────────
│   // The learned clause now has exactly 1 literal at dl (the UIP).
│   // All other literals are at earlier levels.
│   // Backjump target = highest level among those earlier literals.
│   //
│   // After backjumping there:
│   //   - all non-UIP literals are FALSE (their levels are ≤ bt_level)
│   //   - the UIP literal is UNDEF (its level was dl, which was undone)
│   //   → clause is UNIT → propagation forces UIP to TRUE
│
│   if |clause| = 1:
│       bt_level ← 0                      // unit learned clause → go to root
│   else:
│       bt_level ← max{ level[var(l)] : l ∈ clause, level[var(l)] ≠ dl }
│
│   // Bump VSIDS activity for all variables in the learned clause.
│   // This focuses future decisions on the conflict neighborhood.
│   for each var in clause:
│       activity[var] += increment
│   decay all activities periodically (×0.95)
│
│   return (clause, bt_level)
```

### Backjump (non-chronological backtracking)

```
BACKJUMP(target_level):
│
│   // Simply pop the trail until we reach target_level.
│   // All assignments above target_level are erased.
│   // This can skip arbitrarily many decision levels in one shot.
│   //
│   // KEY: watch lists are NOT updated here.
│   // The two-watched-literal scheme is designed so that
│   // backtracking never invalidates the watch invariant.
│   // This makes backjumping O(number of popped literals) with
│   // a very small constant — no clause traversal needed.
│
│   while trail not empty AND level[var(trail.top())] > target_level:
│       lit ← trail.pop()
│       value[lit]  ← UNDEF
│       value[¬lit] ← UNDEF
│       reason[var(lit)] ← NULL
│
│   // After this, the learned clause is unit at target_level.
│   // The next call to PROPAGATE() will pick it up and assign
│   // the asserting literal, driving the search forward.
```

### Control flow

No recursion, no stack frames — one loop, one trail, one flat clause database:

```
    DECIDE → PROPAGATE → OK? ──────────→ next DECIDE
                │                              ↑
             CONFLICT                          │
                │                              │
             ANALYZE                           │
                │                              │
          learn clause                         │
                │                              │
            BACKJUMP ──→ PROPAGATE → OK? ──────┘
                              │
                           CONFLICT → (loop back to ANALYZE)
```

## SMT: Combining SAT with Theories

Chapter 3 lifts SAT to **SMT** (Satisfiability Modulo Theories) via the **DPLL(T)** architecture. The key idea: let a SAT solver handle the Boolean structure; it proposes assignments, and a theory solver checks whether they make sense. If not, the theory solver explains why (a conflict clause), and the SAT solver learns from it. Neither needs to understand the other's internals.

```
┌──────────────────────────────────────────────────┐
│              DPLL(T) / CDCL(T)                   │
│                                                  │
│   ┌────────────┐       ┌─────────────────────┐   │
│   │  SAT Core  │◄─────►│  Theory Solver T    │   │
│   │  (CDCL)    │       │  (e.g. LRA, EUF)    │   │
│   └────────────┘       └─────────────────────┘   │
│        │                        │                │
│        │  assert / check /      │                │
│        │  explain / propagate   │                │
│        ▼                        ▼                │
│   Boolean skeleton         Theory atoms          │
│   (AND/OR/NOT)             (a+b>5, f(x)=y)      │
└──────────────────────────────────────────────────┘
```

DPLL(T) abstracts theory constraints into Boolean variables, uses the SAT core to find a satisfying Boolean assignment, then asks a theory-specific solver whether that assignment is actually feasible in the underlying domain.

### Example

**Problem:** `(x > 0 ∨ x < -5) ∧ (x = -2)`.

1. Abstract to `(A ∨ B) ∧ C`.
2. SAT solver guesses `A=true, C=true` (i.e. `x > 0 ∧ x = -2`).
3. Theory solver checks `x > 0 ∧ x = -2`, returns UNSAT, explains `¬(x > 0 ∧ x = -2)`.
4. SAT solver learns `¬A ∨ ¬C`, backtracks, guesses `B=true, C=true` (`x < -5 ∧ x = -2`).
5. Theory solver returns UNSAT, learns `¬B ∨ ¬C`.
6. SAT solver proves UNSAT.

### Pseudocode

```
DPLL(T)-SOLVE(φ):
    B := abstract(φ)          // replace theory atoms with Boolean vars
    loop:
        α := SAT-SOLVE(B)     // find propositional model
        if α = UNSAT: return UNSAT

        Γ := {theory literals true under α}
        if T-CONSISTENT(Γ):
            return SAT(α)
        else:
            lemma := T-EXPLAIN(Γ)   // subset of Γ that is T-unsat
            B := B ∧ ¬lemma         // block this combination
```

The theory solver exposes a small incremental interface, and all CDCL machinery (two-watched literals, VSIDS, backjumping) carries over unchanged:

| Method | Purpose |
|:---|:---|
| `T-ASSERT(lit)` | Incrementally add a theory literal |
| `T-CHECK()` | Return SAT/UNSAT for the current conjunction |
| `T-EXPLAIN()` | Return a minimal T-unsatisfiable subset (conflict clause) |
| `T-PROPAGATE()` | Deduce implied literals, feed back to the SAT core |

## Equality Logic and Uninterpreted Functions

Chapter 4 handles the simplest theory: equality (`=`, `≠`) over variables from an infinite domain, extended with **uninterpreted functions** whose only axiom is **functional consistency** (congruence): equal arguments produce equal results, `(⋀ᵢ tᵢ = tᵢ′) ⟹ F(t̄) = F(t̄′)`.

The decision procedure is **congruence closure**: group terms into equivalence classes from the explicit equalities, merge classes whose terms are the same function applied to already-equal arguments, then check that no disequality lands inside a single class.

```
CONGRUENCE-CLOSURE(φ):
    // φ is a conjunction of equalities and disequalities over UF terms

    1. Build initial classes from equalities:
       for each (t₁ = t₂) in φ: merge(class(t₁), class(t₂))

    2. Transitivity closure:
       while ∃ classes with a shared term: merge them

    3. Congruence closure:
       while ∃ F(tᵢ), F(tⱼ) in φ with tᵢ ≡ tⱼ (same class):
           merge(class(F(tᵢ)), class(F(tⱼ)))

    4. Check disequalities:
       if ∃ (tᵢ ≠ tⱼ) in φ with tᵢ, tⱼ in the same class: return UNSAT
       return SAT
```

With a union-find structure this runs in `O(n log n)`. For `x = y ∧ f(x) ≠ f(y)`: merge `{x, y}`; congruence forces `f(x) ≡ f(y)`; but the disequality demands they differ — UNSAT.

### Application: proving program equivalence

EUF abstracts away the meaning of operators so two programs can be compared structurally. Consider two implementations of cubing:

```c
int power3(int in) {              int power3_new(int in) {
  int i, out_a;                     int out_b;
  out_a = in;                       out_b = (in * in) * in;
  for (i = 0; i < 2; i++)           return out_b;
    out_a = out_a * in;           }
  return out_a;
}
```

Unroll, convert to SSA, and conjoin each side; the verification condition is `in0_a = in0_b ∧ φ_a ∧ φ_b ⟹ out2_a = out0_b`. Replacing `*` with an uninterpreted function `G` turns multiplication into an opaque-but-consistent symbol:

```
φ^UF_a:  out0_a = in0_a  ∧  out1_a = G(out0_a, in0_a)  ∧  out2_a = G(out1_a, in0_a)
φ^UF_b:  out0_b = G(G(in0_b, in0_b), in0_b)
```

Congruence closure chains the equalities through `G` and proves `out2_a = out0_b`. **Valid** — the programs are equivalent, and we never had to reason about multiplication itself.

## Linear Arithmetic

Chapter 5 decides conjunctions of linear constraints. The constraints carve a convex polytope; the right algorithm depends on their shape:

```
                   Your constraints
                        │
          ┌────────────┴────────────┐
          │                         │
    All of the form             General linear
    x - y ≤ c ?                 (a₁x₁ + a₂x₂ + ... ≤ b)?
          │                         │
          ▼                    ┌─────┴─────┐
    ★ Bellman-Ford            │           │
    (negative cycle = UNSAT) Reals?    Integers?
                              │           │
                              ▼           ▼
                        ★ Simplex    ★ Simplex + Branch & Bound
```

| Situation | Use | Must learn? |
|:---|:---|:---|
| General linear over reals (SMT core) | **General Simplex** | **Yes** — the workhorse of every SMT solver |
| General linear over integers | **Simplex + Branch & Bound** | **Yes** — one extra idea on top of Simplex |
| All constraints are `x - y ≤ c` (scheduling, timing) | **Bellman–Ford** | Know it exists; trivial given shortest paths |
| Quantifier elimination (preprocessing) | Fourier–Motzkin | Skip unless eliminating quantifiers |
| Dense integer feasibility (niche) | Omega test | Skip — academic interest |

### General Simplex

Each inequality is a half-space; their conjunction is a convex polytope, and a violated bound always has its worst case at a vertex — so Simplex hops vertex-to-vertex until all bounds hold (SAT) or a row proves no vertex works (UNSAT).

```
        y
        ▲
        │╲  x + y ≤ 2
    2 ──┤  ╲─────────────
        │   ╲ feasible
        │    ╲ region       ← this is the polytope
    1 ──┤     ╲
        │      ╲
    0 ──┼───┬───╲──────► x
        0   1   2
            │
          x ≥ 1
```

The procedure: (1) rewrite inequalities as equations plus bounds (`x + y ≤ 2` becomes `s = x + y` with `s ≤ 2`); (2) split variables into **nonbasic** (knobs you turn freely) and **basic** (readouts fixed by the equations); (3) if all readouts are within bounds → SAT; (4) otherwise **pivot** — find a knob that fixes a violated readout and swap their roles (move to an adjacent vertex); (5) if no helpful knob exists → UNSAT, and the tableau row is a Farkas certificate of infeasibility; (6) **Bland's rule** (always pick the first violating variable) prevents cycling.

For `x + y ≤ 2 ∧ x ≥ 1 ∧ y ≥ 2`: starting at the origin and pivoting to satisfy `x ≥ 1` then `y ≥ 2` drives `s = x + y` to 3, violating `s ≤ 2`, and no pivot can lower `s` without breaking another bound — UNSAT. Simplex is fast in practice because feasible regions are low-dimensional, it stops as soon as feasibility is reached, and adding a bound only tightens one face.

### Branch and Bound

Integer solutions are lattice points inside the polytope. Solve the real relaxation with Simplex; if the optimum is fractional, cut the polytope to exclude the gap and recurse:

```
BRANCH-AND-BOUND(S):
    α := SIMPLEX(relax to reals)      // fast
    if UNSAT: return UNSAT             // prune entire subtree
    if α is all-integer: return SAT    // lucky

    pick variable v with fractional value r
    return BRANCH-AND-BOUND(S ∧ v ≤ ⌊r⌋)
        or BRANCH-AND-BOUND(S ∧ v ≥ ⌈r⌉)
```

Simplex does the heavy lifting; B&B just splits on fractions. As an application, a compiler can hoist a load `a[j]` out of `for(i=1;i<=10;i++) a[j+i]=a[j];` only if the accesses never alias. Checking `i ≥ 1 ∧ i ≤ 10 ∧ j + i = j` yields UNSAT (it requires `i = 0`), so the optimization is safe.

## Bit Vectors

Chapter 6 decides fixed-width machine arithmetic. Every bit-vector operation is a finite circuit, so the theory is always decidable: **bit-blast** the formula into a Boolean circuit (adders, muxes, shifters), flatten to CNF, and call SAT.

```
BV-FLATTENING(φ):
    B := e(φ)                              // propositional skeleton
    for each bit-vector term t[l] in φ:
        allocate l fresh Boolean vars e(t)₀..e(t)_{l-1}
    for each atom a in φ:
        B := B ∧ BV-CONSTRAINT(e, a)       // encode atom
    for each term t in φ:
        B := B ∧ BV-CONSTRAINT(e, t)       // encode operator
    return B                               // pass to SAT solver
```

Each operator expands into its circuit; addition, for instance, is a ripple-carry chain of full adders:

| Operator | Encoding |
|:---|:---|
| Bitwise OR `a \| b` | `e(t)ᵢ ⟺ (aᵢ ∨ bᵢ)` for each bit i |
| Addition `a + b` | Ripple-carry adder (full adder per bit) |
| Subtraction `a - b` | `add(a, ~b, cin=1)` (two's complement) |
| Multiplication | Shift-and-add circuit |
| Comparison `a < b` | Subtraction + sign bit |
| Equality `a = b` | `⋀ᵢ (aᵢ ⟺ bᵢ)` |

```
sum(a, b, cin)   = a ⊕ b ⊕ cin
carry(a, b, cin) = (a ∧ b) ∨ ((a ⊕ b) ∧ cin)
```

This precision catches bugs that integer reasoning misses. The equivalence `(x − y > 0) ⇔ (x > y)` holds over ℤ but **fails** for bit vectors because of overflow — e.g. `unsigned char` `200 + 100` wraps to `44` (the 9th bit is discarded). Bit-blasting encodes the wraparound exactly and the SAT solver reports the equivalence invalid.

## Arrays

Chapter 7 adds arrays, which have just two operations — read `a[i]` and write `a{i←e}` — governed by a few axioms:

| Axiom | Formula | Meaning |
|:---|:---|:---|
| Select | `(a₁=a₂ ∧ i=j) ⟹ a₁[i]=a₂[j]` | Same array, same index → same value |
| Read-over-write | `a{i←e}[j] = (j=i ? e : a[j])` | Write at i, read at j |
| Extensionality | `(∀i. a₁[i]=a₂[i]) ⟹ a₁=a₂` | Element-wise equal → arrays equal |

The decision procedure eliminates arrays entirely, reducing to EUF + the index theory:

```
ARRAY-REDUCTION(φ_A):     // input in NNF; output in index theory + EUF
    1. WRITE RULE: replace each a{i←e} with a fresh array a',
       add a'[i]=e ∧ ∀j≠i. a'[j]=a[j]
    2. Replace ∃i.P(i) with P(j), j fresh
    3. Replace ∀i.P(i) with ⋀_{i∈I(φ)} P(i)   // I(φ) = index expressions in φ
    4. Replace reads a[i] with uninterpreted Fₐ(i)
    return resulting EUF + index-theory formula
```

For `a[i] = 5 ∧ a{i←10}[i] = 5`: the write rule introduces `a'[i] = 10`, so we get `a'[i] = 5 ∧ a'[i] = 10`, i.e. `5 = 10` — UNSAT. The same pipeline discharges array-initialization proofs: the inductive step "given `∀x. x<i ⟹ a[x]=0`, after `a' = a{i←0}` show `∀x. x≤i ⟹ a'[x]=0`" reduces to a finite EUF query over the relevant indices and is closed by congruence closure.

## Pointer Logic

Chapter 8 models memory as an array indexed by addresses, so pointer reasoning needs no new procedure — it reduces to arrays + linear arithmetic. With `L[v]` the address of variable `v` and `M[addr]` the value stored there:

| Expression | Meaning |
|:---|:---|
| `*p` | `M[⟦p⟧]` (dereference = read memory at the pointer's value) |
| `&v` | `L[v]` (address-of) |
| `p + t` | `⟦p⟧ + ⟦t⟧` (pointer arithmetic) |
| `v[t]` | `M[L[v] + ⟦t⟧]` (array access) |
| `s.f` | `*((&s) + offset(f))` (struct field) |

A small memory model keeps it sound: `∀v. L[v] ≠ 0` (nothing at NULL), `∀v. size(v) ≥ 1`, and distinct variables occupy non-overlapping address ranges (separation).

The procedure is **semantic translation**: rewrite the pointer formula into arrays + arithmetic, then hand it to the combined solver. For the valid formula `p = &x ∧ x = 1 ⟹ *p = 1`:

```
⟦p = &x ∧ x = 1 ⟹ *p = 1⟧
= M[L[p]] = L[x]  ∧  M[L[x]] = 1  ⟹  M[M[L[p]]] = 1
```

Substituting `M[L[p]] = L[x]` into `M[M[L[p]]]` gives `M[L[x]] = 1`. **Valid.** Conversely `*p = x ⟹ p = &x` is invalid, and the solver returns a concrete memory layout (e.g. `L[p]=1, L[x]=2, M[1]=3, M[2]=10, M[3]=10`) where the values match but the addresses don't.

## Quantified Formulas

Chapter 9 confronts quantifiers, where decidability is the central question. First-order logic with quantifiers is undecidable in general, but specific fragments are decidable:

| Fragment | Decidable? | Why |
|:---|:---|:---|
| Quantified Boolean (QBF) | Yes (PSPACE-complete) | Finite domain — can always expand |
| Linear arithmetic over ℚ | Yes (doubly exponential) | Fourier–Motzkin eliminates variables |
| Presburger arithmetic (ℤ, +, <) | Yes (triply exponential) | Omega test / automata |
| ℤ with multiplication | **No** | Encodes Diophantine equations |
| Arrays with arbitrary quantifiers | **No** | Undecidable |
| First-order logic (general) | **No** | Halting-problem reduction |

Two strategies apply, depending on whether the theory has a **projection operator** that can eliminate one variable at a time:

```
        ┌─────────────────────────┐
        │ Does the theory admit   │
        │ a projection operator?  │
        └───────────┬─────────────┘
              │
     Yes ┌────┴────┐ No
         ▼              ▼
  Quantifier        Heuristic
  Elimination       Instantiation
  (complete,        (incomplete,
   always halts,     may not
   expensive)        terminate)
```

**Quantifier elimination** converts to prenex form and eliminates innermost-first: `∃x.ψ` becomes `PROJECT(ψ, x)`, and `∀x.ψ` becomes `¬PROJECT(¬ψ, x)`. QBF projects by Shannon expansion, ℚ by Fourier–Motzkin, Presburger by the Omega test. The catch is blow-up — Fourier–Motzkin pairs every lower bound with every upper bound, squaring the constraint count per eliminated variable.

When no projection operator exists, fall back to **heuristic instantiation**: Skolemize, then repeatedly instantiate universal axioms with ground terms already in the formula, using **trigger-based E-matching** (pattern matching modulo the congruence/E-graph) to pick substitutions, checking each ground result with DPLL(T). For example, to prove `f(h(a), b) = f(b, h(a))` from `∀x∀y. f(x,y) = f(y,x)`, the trigger `f(x,y)` matches `f(h(a), b)`, instantiates the commutativity axiom, and DPLL(T) closes the goal. This is incomplete — new instances can spawn new terms forever — so solvers use cycle detection and user-supplied triggers to curb divergence.

| | QE (Approach 1) | Instantiation (Approach 2) |
|:---|:---|:---|
| Complete | Yes | No |
| Terminates | Always | Not guaranteed |
| Applies to | Theories with projection | Any (heuristic) |
| Used for | QBF, linear arithmetic | Verification conditions, axiomatized theories |

A striking QE application is encoding "can White force checkmate in `k` moves?" as a QBF: existentials for White's moves, universals for Black's, an implication ensuring White need only answer **legal** Black replies, and a goal `G_k` capturing checkmate.

## Combining Theories: Nelson–Oppen

Real verification conditions mix theories (arithmetic + uninterpreted functions + arrays). Chapter 10's **Nelson–Oppen** method combines decision procedures by exchanging only **equalities between shared variables** — ordering, arithmetic, and array content all stay internal to each theory. It applies when the theories are quantifier-free, signature-disjoint, **stably infinite**, and individually decidable. These conditions are mild: signature-disjointness is achieved mechanically by **purification**, and almost every practical theory (LRA, EUF, arrays) is stably infinite (bit-vectors, being finite, need special handling).

Purification replaces alien subterms with fresh variables so each conjunct lives in one theory:

```
x₁ ≤ f(x₁)  ──purify──→  x₁ ≤ a  ∧  a = f(x₁)
                           ─────────    ───────────
                           arithmetic      EUF
```

For **convex** theories (LRA, EUF) pure equality propagation suffices:

```
NELSON-OPPEN-CONVEX(φ):
    1. Purify φ into pure conjuncts F₁, ..., Fₙ (one per theory)
    2. For each Fᵢ: if Tᵢ-UNSAT(Fᵢ) → return UNSAT
    3. Equality propagation:
       if Fᵢ ⊨ (x = y) and Fⱼ ⊭ (x = y): add (x = y) to Fⱼ; goto 2
    4. return SAT
```

**Nonconvex** theories (integers, bit-vectors) can imply a disjunction of equalities without implying any single one (e.g. `1 ≤ x ≤ 2` forces `x=1 ∨ x=2`), so standalone Nelson–Oppen must **case-split**, which is exponential. The elegant fix inside DPLL(T): push the disjunction `x=1 ∨ x=2` back to the SAT engine as a **splitting clause** — nonconvexity becomes just another Boolean decision handled natively by CDCL.

```
┌─────────────────────────────────────────────────────┐
│  DPLL(T) — outer loop (Boolean reasoning)            │
│    SAT engine picks assignment → THEORY-CHECK        │
│    ┌──────────────────────────────────────────────┐ │
│    │  THEORY-CHECK (Nelson–Oppen inside)          │ │
│    │    1. Purify                                 │ │
│    │    2. Send pieces to theory solvers          │ │
│    │    3. Propagate equalities                   │ │
│    │    4. If nonconvex split needed:             │ │
│    │       push (x=a ∨ x=b) back to SAT as clause │ │
│    └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

For `x ≤ 1 ∧ x ≥ 0 ∧ f(x) ≠ f(0) ∧ f(x) ≠ f(1)` over integers + EUF, arithmetic deduces `x=0 ∨ x=1`; the SAT engine tries each branch, congruence derives the matching equality, and both branches conflict — UNSAT.

## Propositional Encodings of EUF

When a formula's only theory content is equality and uninterpreted functions, Chapter 11 shows you can skip the theory solver entirely: replace each function call with a fresh variable, add "equal arguments imply equal results" constraints, and hand a single monolithic SAT problem to CDCL.

**Ackermann's reduction** adds the consistency constraints pairwise:

```
ACKERMANN(φ^UF):     // m instances of uninterpreted function F
    1. Index UF instances F₁..Fₘ from subexpressions outward
    2. Flatten: replace each Fᵢ with fresh variable fᵢ → flat^E
    3. FC^E := ⋀_{1≤i<j≤m} (arg(Fᵢ)=arg(Fⱼ) ⟹ fᵢ=fⱼ)
    4. return FC^E ⟹ flat^E     // validity   (or FC^E ∧ flat^E for sat)
```

This costs `O(m²)` constraints. **Bryant's reduction** instead defines each flattened symbol with an ordered case expression that reuses earlier values, cutting the redundant pairs down to `O(m)` case expressions:

| | Ackermann | Bryant |
|:---|:---|:---|
| Constraints | `O(m²)` pairwise implications | `O(m)` case expressions |
| Redundancy | High | Lower (ordered reuse) |
| Implementation | Simpler | More complex |

These encodings shine for hardware and compiler-equivalence checking, where operations are abstracted as uninterpreted functions — but they blow up for thousands of function instances, where congruence closure inside DPLL(T) is usually faster.

## Software Verification

Chapter 12 turns the theory machinery on real programs. (This is the SMT-based counterpart to the [Software Model Checking](#software-model-checking) of the first part: there the program became a Boolean program checked with abstraction; here it becomes a *formula* checked with a decision procedure.) The obstacle is that programs are dynamic while decision procedures are static, and verification is undecidable in general (unbounded loops, unbounded memory). The bridge is **SSA** (Static Single Assignment): rename every variable write into a fresh timestamped symbol, and an entire program path collapses into a conjunction of equalities and conditions — a quantifier-free formula for an SMT solver.

Two practical strategies trade completeness for decidability, and both end in "is this QF formula satisfiable?":

```
┌──────────────────────┐       ┌──────────────────────────────┐
│  BOUNDED ANALYSIS    │       │  UNBOUNDED ANALYSIS          │
│  (underapproximation)│       │  (overapproximation)         │
│                      │       │                              │
│  • Unroll loops k    │       │  • Replace loops with        │
│    times             │       │    nondeterministic assigns  │
│  • SSA → formula     │       │  • Add assume/assert for     │
│  • SAT ⟹ real bug   │       │    loop invariants           │
│  • UNSAT ⟹ safe     │       │  • SAT ⟹ spurious or bug   │
│    (up to bound k)   │       │  • UNSAT ⟹ safe for ALL     │
│                      │       │    iterations                │
└──────────────────────┘       └──────────────────────────────┘
```

### Translating programs into formulas (SSA)

Rename every variable at each assignment with a fresh timestamped symbol, then read the program off as a conjunction:

| Program construct | SSA translation |
|:---|:---|
| Assignment `x = expr` | Introduce fresh `xₖ₊₁ = expr` (using current timestamps) |
| Read variable `x` | Use the most recent timestamp `xₖ` |
| Branch taken (condition `c`) | Add `c` as a conjunct |
| Branch not taken | Add `¬c` as a conjunct |
| Assertion `assert(P)` | Add `¬P` as a conjunct (to check for a violation) |

For a chosen path through a loop, timestamping every write and conjoining the assignments and branch conditions yields a single **path constraint**:

```
ssa ≡  i₁ = 0
     ∧ next₁ = data₀[i₁]
     ∧ (i₁ < next₁ ∧ next₁ < N₀)
     ∧ i₂ = i₁ + 1
     ∧ i₂ < next₁
     ∧ data₀[i₂] ≠ cookie₀
     ∧ i₃ = i₂ + 1
     ∧ ¬(i₃ < next₁)
     ∧ next₂ = data₀[i₃]
     ∧ ¬(i₃ < next₂ ∧ next₂ < N₀)
```

To check whether `assert(0 ≤ i ∧ i < N)` can be violated on this path, conjoin the path constraint with the **negated** assertion: `VC ≡ ssa ∧ ¬(0 ≤ i₄ ∧ i₄ < N₀)`. **SAT** means a real bug — and the satisfying assignment *is* the triggering input (e.g. `data₀ ↦ ⟨2, 6, 5⟩, N₀ ↦ 3`, walking `i` off the end of the array). **UNSAT** means the assertion holds on that path.

### Bounded analysis: all paths in one formula

Enumerating paths one at a time is exponential. Instead, **unroll** each loop `k` times, assign every branch condition to a fresh **guard** variable `γ`, and at each merge point use a **φ-instruction** — a conditional `γ ? then_value : else_value` — so a single formula encodes all `2ⁿ` paths at once:

```
γ₁ = (i₀ < next₀)
γ₂ = (data₀[i₁] == cookie₀)
i₁ = i₀ + 1
i₂ = γ₂ ? i₁ : i₀          ← φ: merge after inner if
i₃ = i₂ + 1
γ₃ = (i₃ < next₀)
γ₄ = (data₀[i₃] == cookie₀)
i₄ = i₃ + 1
i₅ = γ₄ ? i₄ : i₃          ← φ: merge after inner if
i₆ = i₅ + 1
i₇ = γ₃ ? i₆ : i₃          ← φ: merge after second unrolling
i₈ = γ₁ ? i₇ : i₀          ← φ: merge after outer if
```

The solver explores all path combinations by choosing values for the `γ`s; there is no explicit path enumeration. This is **underapproximation** — sound for bugs up to depth `k`, but it says nothing beyond the unrolling bound.

### Unbounded analysis: overapproximation + loop invariants

To reason about *all* iterations, replace each loop by a single havoc-and-iterate step: set every loop-modified variable to a nondeterministic value `*`, run the body once, and `assume(¬C)` after. So `while(C){B}` becomes `if(C){ var=*; B }; assume(¬C)`. For a loop that copies `i` into `j`, the resulting formula is UNSAT against `i ≠ j`, proving `i == j` for **any** number of iterations.

Pure havoc is often **too coarse**, producing spurious counterexamples. In a lock protocol, `state_of_lock = *` at loop entry lets the solver enter the body already `locked`, violating `assert(state_of_lock == unlocked)` — a path that cannot occur in the real program. The fix is to supply a **loop invariant** `I` and discharge it by induction, woven into one loop-free program:

```
┌─────────────────────────────────────────────────────────┐
│  Template:   A; while(C) { assert(I); B; }              │
├─────────────────────────────────────────────────────────┤
│  BASE CASE:        A;                                    │
│                    assert(C ⟹ I);                        │
│                                                          │
│  STEP CASE:        assume(C ∧ I);                        │
│                    B;                                     │
│                    assert(C ⟹ I);                        │
│                                                          │
│  Both UNSAT ⟹ I is a valid loop invariant               │
└─────────────────────────────────────────────────────────┘
```

`assume(I)` after the havoc encodes the induction hypothesis (restricting the otherwise-arbitrary state), the `assert`s give the base and step cases, and the property assertions in the body ride on top. With invariant `state_of_lock == unlocked`, every check in the lock example discharges to UNSAT, proving the protocol correct for all iterations.

### Proof organization: many assertions, many VCs

A subtlety: the base case and step case are checked with one *program text*, but **not** one formula with all negated assertions ANDed together. `assume(P)` adds `P` as a conjunct, but each `assert(Q)` spawns its **own** verification condition:

```
VC_k = (all assumes and assignments up to assert_k) ∧ ¬(assert_k)
       UNSAT → assertion k holds;   SAT → counterexample
```

You cannot negate several assertions at once — a trivially-UNSAT negation would mask failures in the others. The lock example thus generates five independent VCs (base case, step case, and three property checks), and all must be UNSAT.

| Statement | Role | Effect in VC |
|:---|:---|:---|
| `assert(I)` before havoc | Base case | VC: prefix ∧ ¬I |
| `assume(I)` after havoc | Induction hypothesis | Added as a conjunct in all later VCs |
| `assert(C ⟹ I)` at loop end | Step case | VC: prefix ∧ ¬(C ⟹ I) |
| `assert(property)` in body | Property check | VC: prefix ∧ ¬property |

Each loop gets its own invariant and havoc block; nested loops are handled inside-out, transforming the inner loop to a loop-free fragment before the outer one. The remaining hard problem is **invariant selection** — there is no algorithm that always finds an invariant strong enough, which is exactly where techniques like CEGAR and IC3 from the model-checking part come back into play.

## The Decision Pipeline

Putting the chapters together, an arbitrary formula flows through preprocessing and dispatch into the right engine:

```
Input formula φ (arbitrary theory mix, quantifiers, Boolean structure)
    │
    ├─ Has quantifiers? ──→ Prenex NF → Quantifier Elimination (Ch.9)
    │                           └─ QBF: resolution/expansion
    │                           └─ LRA: Fourier–Motzkin
    │                           └─ LIA: Omega test
    │
    ├─ Mixed theories? ──→ Nelson–Oppen (Ch.10)
    │                           └─ Purify → local solve → propagate equalities
    │
    ├─ Boolean structure? ──→ DPLL(T) / CDCL(T) (Ch.3)
    │                           └─ SAT core + theory solver interface
    │
    └─ Pure conjunctive theory fragment:
         ├─ EUF         → Congruence Closure (Ch.4)
         ├─ LRA         → General Simplex (Ch.5)
         ├─ LIA         → Branch & Bound (Ch.5)
         ├─ Difference  → Bellman–Ford (Ch.5)
         ├─ Bit-vectors → Bit-blast to SAT (Ch.6)
         ├─ Arrays      → Reduce to EUF + index theory (Ch.7)
         └─ Pointers    → Reduce to integers + arrays (Ch.8)
```

## Complexity

Adding Boolean structure on top of a conjunctive fragment generally pushes the problem to NP-complete (the SAT core is the source of hardness):

| Theory / Problem | Conjunctive fragment | Full Boolean structure |
|:---|:---|:---|
| Propositional (SAT) | — | NP-complete |
| Equality logic | Polynomial | NP-complete |
| EUF | $O(n \log n)$ | NP-complete |
| LRA (reals) | Polynomial (Simplex) | NP-complete (via DPLL(T)) |
| LIA (integers) | NP-complete | NP-complete |
| Difference logic | $O(\|V\|\cdot\|E\|)$ | NP-complete |
| Bit-vectors | Reduces to SAT | NP-complete |
| Arrays (property fragment) | Reduces to EUF | NP-complete |
| QBF | — | PSPACE-complete |
| Presburger arithmetic | — | $2^{2^{cn}}$ lower bound |

## Z3: An SMT Solver End to End

Modern SMT solvers like **Z3** are, at heart, the DPLL(T)/CDCL(T) engine of Chapter 3 with every other chapter plugged in as a module. A CDCL SAT engine handles all Boolean reasoning; **EUF / congruence closure** is the central hub through which theories exchange equalities; arithmetic is a dual Simplex + branch-and-cut solver; bit-vectors are bit-blasted into SAT; arrays are reduced to EUF by lazy axiom instantiation; pointers reduce to arrays + integers; quantifiers are handled by E-matching and model-based instantiation; theory combination is model-based Nelson–Oppen; and propositional (Ackermann) encodings are injected selectively.

```
┌──────────────────────────────────────────────────────────────┐
│                     USER QUERY (SMT-LIB2)                     │
└───────────────────────────────┬──────────────────────────────┘
                          ┌──────▼──────┐
                          │   TACTICS   │  ← preprocessing pipeline
                          │ (simplify,  │    (rewriting, macro-finding,
                          │  solve-eqs, │     quantifier elimination,
                          │  qe-light)  │     bit-blasting decisions)
                          └──────┬──────┘
                  ┌──────────────▼──────────────┐
                  │       ENGINE DISPATCHER      │
                  └──┬──────────┬──────────┬─────┘
            ┌────────▼──┐  ┌───▼────┐  ┌──▼──────┐
            │  CDCL(T)  │  │ NLSAT  │  │ SPACER  │
            │ (default) │  │(MC-SAT)│  │  (CHC)  │
            └───────────┘  └────────┘  └─────────┘
```

The default path — `SAT engine (CDCL) ↔ EUF hub ↔ {arithmetic, arrays, bit-vectors, …}` — is exactly the DPLL(T) architecture, scaled up with engineering. Specialized engines take over for fragments where CDCL(T) is not ideal: **NLSAT** for nonlinear real arithmetic (CAD + model-constructing search), **SPACER** for constrained Horn clauses (IC3-style, connecting back to the [PDR](#sat-based-model-checking) of the model-checking part), and **QSAT** for quantified formulas over theories with projection.

What makes Z3 fast beyond the textbook algorithm is mostly throttling and laziness: **relevancy filtering** (only expose relevant literals to expensive theory solvers), **dynamic Ackermann reduction** (inject congruence/transitivity short-cuts when conflicts indicate benefit), **model-based theory combination** (propose shared equalities from the current model instead of enumerating them), an **incremental nonlinear-arithmetic waterfall** (bounds → Horner → Gröbner → linearization → full NLSAT, paying for CAD only when forced), and **code-tree E-matching** (quantifier patterns compiled into a backtrackable bytecode VM).

# Automated Reasoning

Decision procedures decide *quantifier-free* formulas. Building a complete verification system needs higher-level reasoning on top: a logic expressive enough to state program properties, a calculus for proving validity, and frameworks that translate programs into logic and discharge the result with an SMT solver. This part follows UT Austin's [CS 389L — Automated Logical Reasoning](https://www.cs.utexas.edu/~isil/cs389L/) (Işıl Dillig), covering first-order logic, resolution theorem proving, Hoare logic, VC generation, and abstract interpretation.

## First-Order Logic

First-order logic (FOL) extends propositional logic with quantifiers, functions, and predicates over a universe of objects, so it can describe infinite structures that truth tables cannot:

| Component | Propositional | First-order |
|:---|:---|:---|
| Atoms | Boolean variables `p, q, r` | Predicates over terms: `loves(x, mother(y))` |
| Constants | `⊤, ⊥` | Object/function/relation constants |
| Quantifiers | none | `∀x`, `∃x` |
| Expressiveness | finite truth tables | properties of infinite structures |

There are three kinds of symbols: **object constants** name specific objects (`joe`, `3`), **function constants** map objects to objects (`mother(x)`, `plus(x,y)`; an object constant is just arity-0), and **relation constants (predicates)** express properties and return true/false (`loves(x,y)`, `isPrime(n)`). A crucial syntactic rule: functions nest freely inside predicates and other functions (`p(f(g(x)))` is fine), but predicates never nest — `f(p(x))` and `p(q(x))` are not FOL.

Meaning requires a **structure** `S = ⟨U, I⟩` (a non-empty universe `U` and an interpretation `I` fixing every constant) plus a **variable assignment** `σ`:

```
Terms:       ⟨I,σ⟩(a) = I(a),  ⟨I,σ⟩(x) = σ(x),
             ⟨I,σ⟩(f(t₁,...,tₖ)) = I(f)(⟨I,σ⟩(t₁),...,⟨I,σ⟩(tₖ))
Predicates:  S,σ ⊨ p(t̄) iff ⟨...⟩ ∈ I(p)
Quantifiers: S,σ ⊨ ∀x.F iff S,σ[x↦o] ⊨ F for ALL o ∈ U
             S,σ ⊨ ∃x.F iff S,σ[x↦o] ⊨ F for SOME o ∈ U
```

A small evaluation, with `U = {●, ★}`, `I(b) = ★`, `I(f) = {● ↦ ★, ★ ↦ ●}`, `I(p) = {⟨★,●⟩, ⟨★,★⟩}`, `σ = {x ↦ ●}`:

```
p(f(b), f(x)):  f(b)=●, f(x)=★ → is ⟨●,★⟩ ∈ I(p)? No  → FALSE
∀x. p(b, x):    x↦●: ⟨★,●⟩∈I(p)? Yes;  x↦★: ⟨★,★⟩∈I(p)? Yes  → TRUE
```

As in propositional logic, `F` is **satisfiable** if some structure-and-assignment models it, **valid** if all do, and the duality `F valid ⟺ ¬F unsatisfiable` still holds — the foundation of refutation-based proving.

## Properties of First-Order Logic

The bad news: full FOL validity is **undecidable** (Church, Turing). The good news: it is **semi-decidable** — if a formula is valid, a complete proof system eventually finds a proof; if it is not, the prover may run forever. Restricted fragments recover decidability:

| Fragment | Status |
|:---|:---|
| Full FOL validity | undecidable, but semi-decidable |
| Quantifier-free FOL | decidable (NP-complete) — what decision procedures handle |
| Monadic FOL (unary predicates, no functions) | decidable |
| Bernays–Schönfinkel (`∃*∀*` prefix, no functions) | decidable |
| FOL with a single binary predicate | undecidable |

The **compactness theorem** (an infinite set of sentences is satisfiable iff every finite subset is) is the standard tool for showing something is *not* expressible in FOL. For instance, transitive closure is inexpressible: if `Γ` encoded "T is the transitive closure of p," then `Γ ∪ {T(a,b)} ∪ {Ψₙ : "no path of length n from a to b"}` has every finite subset satisfiable (pick a path longer than any `n` present), so by compactness the whole set is satisfiable — yet it is plainly contradictory.

To prove validity directly, the **semantic argument method** assumes `S,σ ⊭ F` and applies proof rules until every branch reaches a contradiction. The quantifier rules hinge on *fresh* vs. *arbitrary* objects:

```
S,σ ⊨ ∀x.F  →  S,σ[x↦o] ⊨ F  for ANY o      S,σ ⊭ ∀x.F  →  ... ⊭ F  for a FRESH o
S,σ ⊨ ∃x.F  →  S,σ[x↦o] ⊨ F  for a FRESH o   S,σ ⊭ ∃x.F  →  ... ⊭ F  for ANY o
```

## Unification and Clausal Form

Lifting resolution from propositional to first-order logic needs two new ingredients. The first is **unification**: given expressions `e` and `e'`, find a substitution `σ` making them syntactically identical (`eσ = e'σ`).

| Expression 1 | Expression 2 | Unifiable? | MGU |
|:---|:---|:---|:---|
| `p(x, y)` | `p(a, v)` | Yes | `[x ↦ a, y ↦ v]` |
| `p(x, x)` | `p(a, b)` | No | `x` can't be both `a` and `b` |
| `p(x)` | `p(f(x))` | No | occurs check: `x` is inside `f(x)` |
| `p(f(x), f(x))` | `p(y, f(a))` | Yes | `[y ↦ f(a), x ↦ a]` |

The **most general unifier (MGU)** commits to the minimum necessary — every other unifier is a further substitution of it — and is unique up to renaming. Robinson's algorithm computes it recursively, with the **occurs check** preventing a variable from unifying with a term that contains it:

```
find_mgu(e, e'):
    if e = e':            return []
    if e is variable x:   if x occurs in e' → FAIL;  else return [x ↦ e']
    if e' is variable y:  symmetric
    if e = f(e₁..eₖ) and e' = f(e₁'..eₖ'):
        σ := []
        for i in 1..k: σ := σ ∘ find_mgu(eᵢσ, eᵢ'σ)
        return σ
    otherwise:            FAIL
```

The second ingredient is **clausal form** — any FOL formula becomes an *equisatisfiable* set of universally-quantified clauses in five steps:

```
1. Close free variables   (∃-quantify them)
2. Prenex normal form     (NNF + rename, push quantifiers to front)
3. Skolemize              (replace ∃y under ∀x₁..∀xₖ with f(x₁,...,xₖ))
4. CNF                    (distribute ∨ over ∧)
5. Drop ∀                 (remaining variables are implicitly universal)
```

For example `∀y.(p(y) ∧ ¬(∀z.(r(z) → q(y,z,w))))` skolemizes (with `w↦c`, `z↦f(y)`) to the clauses `{p(y)}`, `{r(f(y))}`, `{¬q(y, f(y), c)}`. Skolemization preserves satisfiability, not equivalence — which is exactly what refutation needs.

## First-Order Resolution

To prove `F` valid, negate it, convert `¬F` to clauses, and repeatedly resolve until the empty clause appears. Resolution here is the [propositional rule](#conflict-analysis-first-uip) from the SAT solver, lifted by unifying the complementary literals first:

```
PROVE-VALID(F):
    1. φ := ¬F
    2. convert φ to clausal form
    3. repeat:
         pick clauses C₁ ∋ A, C₂ ∋ ¬B with σ = MGU(A, B)
         resolvent := (C₁\{A} ∪ C₂\{¬B})σ ;  add it
    4. derived {}  → F is VALID
       no new resolvents → F is NOT valid
       (may loop forever — semi-decidable)
```

A refutation of `{happy(x), sad(x)}, {¬sad(y)}, {¬happy(mother(joe))}`: resolving the first two on `sad` (MGU `[x↦y]`) gives `{happy(y)}`; resolving that with the third (MGU `[y↦mother(joe)]`) gives `{}` — unsatisfiable. **Factoring** (collapsing `p(x) ∨ p(y)` to `p(x)` via `[y↦x]`) is needed for completeness. Resolution is **sound** and, with factoring, **complete** for unsatisfiability — but it is *not* a decision procedure: on a satisfiable set it may generate clauses forever.

## Hoare Logic

Hoare logic reasons about programs compositionally through **triples** `{P} S {Q}`: if precondition `P` holds and `S` terminates, postcondition `Q` holds afterward. One inference rule per construct:

| Construct | Rule |
|:---|:---|
| Assignment | `{Q[E/x]} x := E {Q}` — substitute `E` for `x` in the postcondition |
| Composition | `{P} S₁ {Q}` and `{Q} S₂ {R}` give `{P} S₁; S₂ {R}` |
| Conditional | `{P∧C} S₁ {Q}` and `{P∧¬C} S₂ {Q}` give `{P} if C then S₁ else S₂ {Q}` |
| Loop | `{I∧C} S {I}` gives `{I} while C do S {I∧¬C}` — `I` is the **loop invariant** |
| Consequence | strengthen the precondition (`P'⇒P`) or weaken the postcondition (`Q⇒Q'`) |

The assignment rule extends to arrays via the read-over-write axioms of the [array theory](#arrays): proving `{v[i]=3} v[1]:=2 {v[i]=3}` reduces to `v⟨1↦2⟩[i] = 3`, i.e. `(i=1 ⇒ 2=3) ∧ (i≠1 ⇒ v[i]=3)`.

## Verification Conditions

Doing Hoare proofs by hand is tedious; **VC generation** mechanizes it. For loop-free code the exact weakest precondition `wp(S, Q)` is computable; loops need annotated invariants, so we use an approximate weakest precondition `awp` and emit side conditions:

```
awp(while C do [I] S, Q) ≡ I
VC(while C do [I] S, Q)  ≡ (I ∧ C ⇒ awp(S,I) ∧ VC(S,I)) ∧ (I ∧ ¬C ⇒ Q)
awp(assert(E), Q) = E ∧ Q          awp(assume(E), Q) = E ⇒ Q
```

To prove `{P} S {Q}`, ask an SMT solver whether `P ∧ VC(S,Q) ⇒ awp(S,Q)` is valid; function calls are encoded as `assert(Pre); assume(Post)`. For a loop `while i ≤ n do [sum ≥ 0] { sum:=sum+i; i:=i+1 }` with postcondition `sum ≥ 0`, the generator emits a preservation obligation `(sum ≥ 0 ∧ i ≤ n) ⇒ (sum+i ≥ 0)` and an exit obligation `(sum ≥ 0 ∧ i > n) ⇒ sum ≥ 0`. As in the [software-verification](#software-verification) chapter of the previous part, each assertion is a separate, independently-checked VC.

## Abstract Interpretation and CEGAR

VC generation still needs a human to supply loop invariants. **Abstract interpretation** discovers them automatically by running the program over an abstract domain: pick a lattice of properties (intervals `[a,b]`, signs, octagons `±x±y ≤ c`), define how each statement transforms abstract values, then iterate the control-flow graph — joining states at merge points — to a least fixed point. In infinite-height domains a **widening** operator `∇` extrapolates growing bounds to infinity to force termination (a later narrowing pass can recover precision). For `x=1; while(*) x=2;`, the loop value goes `[1,1] → [1,1]⊔[2,2] = [1,2]`, and widening jumps the moving bound to `[1,∞]`.

When an abstraction is too coarse and reports a false bug, **CEGAR** — the same loop introduced in the [abstraction](#abstraction) section of the model-checking part — refines it. The new element here is *how* the refinement is computed: extract the spurious trace, encode it as a formula and check satisfiability ([UNSAT ⇒ spurious](#software-verification)), then use [Craig interpolation](#sat-based-model-checking) on the UNSAT proof to mine new predicates that exclude the trace. For

```c
x := 0; y := 0;
while (x < 100) { x++; y++; }
assert(y == 100);
```

a coarse predicate set reports a spurious "execute the loop zero times" trace; its formula `x₀=0 ∧ y₀=0 ∧ x₀≥100 ∧ y₀≠100` is UNSAT, and interpolation yields the predicate `x = y ∧ x ≤ 100`, after which the proof goes through. This closes the loop opened back in the first part: the same interpolation that drives SAT-based model checking also supplies the missing invariants for abstraction-based software verification.
