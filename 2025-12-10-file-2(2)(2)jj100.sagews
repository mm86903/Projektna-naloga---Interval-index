︠fce3ddfd-2762-4a2a-87b7-3ccacb7d4926︠
# This worksheet was converted from a notebook running Jupyter kernel
# version sage-10.7.
︡cdbbccb9-36a0-41a6-b353-d1c49cc4425b︡{"stdout": "", "done": true}︡
︠8f2f7b21-426d-4139-9415-1ff1b739e8d8︠
# load previously generated graphs
import csv

def load_results(filename):
    results = []   # list of (Int(G), Radius)

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            IntG   = int(row["Ind(G)"])
            Radius = int(row["Radius"])
            results.append((IntG, Radius))

    return results

n=18

results = load_results(f"C{n}.csv")

all_cubic  = load("C18_cubic_graphs_n10.sobj")
print("Load complete")
︡196c55e5-9d50-4eab-af7d-74b17789cd1d︡{"html": "<pre><span style=\"font-family:monospace;\">Load complete\n</span></pre>", "done": true}︡
︠e4b47e4e-37a6-4f13-a5aa-8e345a193bfc︠
def interval_index(G):
    V = G.vertices()  # dobimo vsa vozlišča
    Int_G = 0
    
    # Iteriramo cez  vse pare vozlisc {u, v}
    for u in V:
        for v in V:
            # Izognemo se dvojnemu stetju (u,v) in (v,u) in (u,u)!
            if u >= v:
                continue
            d_uv = G.distance(u, v)    # poiscemo razdaljo
            
            interval_size = 0   # stevec za |I(u,v)|
            
            for w in V:
                d_uw = G.distance(u, w)
                d_wv = G.distance(w, v)
                
                if d_uw + d_wv == d_uv:
                    interval_size += 1
            
            Int_G += (interval_size - 1)
            
    return Int_G
︡07f0236f-f5ce-4c09-8a04-5aa233dd626d︡{"stdout": "", "done": true}︡
︠f8d0fcf9-5a4b-4211-825c-81fd1fcda989︠
from sage.all import *

deg = 3
n = 16 

all_cubic = list(graphs.nauty_geng(f"{n} -c -d3 -D3"))

save(all_cubic, f"C{n}_cubic_graphs_n10.sobj")
print(f"File savedC{n}_cubic_graphs_n10.sobj")

results = []
for G in all_cubic:
    if G.order() == n and G.is_regular(3):  # Verify cubic
        func_val = interval_index(G)
        diam = G.diameter()
        results.append([func_val, diam])

num_graphs = len(results)
print(f"There are {num_graphs} non isomorphic cubic graphs on {n} vertices.")
print(f"Number of vertices: {n}")
print(f"Number of generated graphs: {num_graphs}")

Gpts = point(results, color = 'lightblue' , size=20)
labels = sum(
    text(str(i+1), (results[i][0], results[i][1]), color='black', fontsize=5)
    for i in range(len(results))
)

p = Gpts + labels
p.axes_labels(['Int(G)', 'diameter'])
p.show() 


#########################

# Plot points
#points = point(results, rgbcolor=(0,0,1), size=50, title=f"Cubic graphs on {n} vertices: Int(G) vs diameter")
#points.show()
︡a5a05f6f-9129-420f-913d-fe67f820953a︡{"html": "<pre><span style=\"font-family:monospace;\">File savedC16_cubic_graphs_n10.sobj\n</span></pre><br/><pre><span style=\"font-family:monospace;\">There are 4060 non isomorphic cubic graphs on 16 vertices.\nNumber of vertices: 16\nNumber of generated graphs: 4060\n</span></pre>", "done": true}︡
︠6091a301-9884-4d43-91b1-1628e4a9d478i︠
%md
**N = 4**

Obstaja en neizomorfen kubičen graf G na 4 vozliščih.

Int\(G\) = 6, radij grafa je 1.

Int\(pot4\) = 10.

**N = 6**

Obstajata dva neizomorfna kubična gafa na 6 vozliščih.

Oba imasta radij 2, Int\(G\) grafov je 27 in 33. Večji Int\(G\) ima graf, ki je bipartiten in ima večji Aut\(G\).

Int\(pot6\) = 35.

**N = 8**

Obstaja 5 neizomorfnih kubičnih gafov G na 8 vozliščih.

Dva grafa imasta radij 2, prvi ima Int\(G\) = 49, drugi pa Int\(G\) = 52.

Trije graf imajo radij 3. Int\(G\) pa 58, 68 in 76.

Edini največji Int\(G\) ima edini bipartitni graf, ki ima tudi največji Aut\(G\). 

Ostali grafi niso bipartitni in imajo večji Int\(G\) pri večjem Aut\(G\).

In imajo pri istem diametru večji Int\(G\) in Aut\(G\).

**N = 10**

Obstaja 19 neizomorfnih kubičnih gafov na 10 vozliščih. Dva sta bipartitna, vsi ostali pa ne.

Int\(pot10\) = 165.

Graf z najmanjšim Int\(G\) ima največji Aut\(G\) in hkrati najmanši radij.

Edini, najvišji radij 5, ima graf z najvišjim Int\(G\). 

Večina \(80%\) grafov ima radij 3.

**N = 12**

Obstaja 85 neizomorfnih kubičnih grafov na 12 vozliščih.

Int\(pot12\) = 286.

Radij grafov sega od 3 pa do 6. Večina ima radij ali 3 ali 4. Od 85 grafov imata dva radij 6, pet jih ima radij 5.

Skoraj noben graf ni bipartiten.

Int\(G\) pada, ko pada tudi radij.

Pri n=12 ima bipartitnost večji vpil kot globina grafa. 

**N =14**

Obstaja 509 neizomorfnih kubičnih grafov na 14 vozliščih.

Int\(pot14\) = 455.

Z večjim radijem, se lepo veča tudi indeks.

Prileganje radija in iintervalnega ndeksa je zelo naravno. Večji radij, da višji indeks.

Najpogostejši radij je 4. Odstopanja niso velika.

Bipartitni grafi so zelo redki, imajo pa najvišji Int\(G\).

**N = 16**

Obstaja 4060 neizomorfnih kubičnih grafov na 16 vozliščih.

Int\(pot16\) = 

Prileganje radija in iintervalnega ndeksa je zelo naravno. Stopničasto.
︡918fa346-cab9-49e7-8919-d2a3e7aaf83a︡{"md": "**N = 4**\n\nObstaja en neizomorfen kubi\u010den graf G na 4 vozli\u0161\u010dih.\n\nInt\\(G\\) = 6, radij grafa je 1.\n\nInt\\(pot4\\) = 10.\n\n**N = 6**\n\nObstajata dva neizomorfna kubi\u010dna gafa na 6 vozli\u0161\u010dih.\n\nOba imasta radij 2, Int\\(G\\) grafov je 27 in 33. Ve\u010dji Int\\(G\\) ima graf, ki je bipartiten in ima ve\u010dji Aut\\(G\\).\n\nInt\\(pot6\\) = 35.\n\n**N = 8**\n\nObstaja 5 neizomorfnih kubi\u010dnih gafov G na 8 vozli\u0161\u010dih.\n\nDva grafa imasta radij 2, prvi ima Int\\(G\\) = 49, drugi pa Int\\(G\\) = 52.\n\nTrije graf imajo radij 3. Int\\(G\\) pa 58, 68 in 76.\n\nEdini najve\u010dji Int\\(G\\) ima edini bipartitni graf, ki ima tudi najve\u010dji Aut\\(G\\). \n\nOstali grafi niso bipartitni in imajo ve\u010dji Int\\(G\\) pri ve\u010djem Aut\\(G\\).\n\nIn imajo pri istem diametru ve\u010dji Int\\(G\\) in Aut\\(G\\).\n\n**N = 10**\n\nObstaja 19 neizomorfnih kubi\u010dnih gafov na 10 vozli\u0161\u010dih. Dva sta bipartitna, vsi ostali pa ne.\n\nInt\\(pot10\\) = 165.\n\nGraf z najmanj\u0161im Int\\(G\\) ima najve\u010dji Aut\\(G\\) in hkrati najman\u0161i radij.\n\nEdini, najvi\u0161ji radij 5, ima graf z najvi\u0161jim Int\\(G\\). \n\nVe\u010dina \\(80%\\) grafov ima radij 3.\n\n**N = 12**\n\nObstaja 85 neizomorfnih kubi\u010dnih grafov na 12 vozli\u0161\u010dih.\n\nInt\\(pot12\\) = 286.\n\nRadij grafov sega od 3 pa do 6. Ve\u010dina ima radij ali 3 ali 4. Od 85 grafov imata dva radij 6, pet jih ima radij 5.\n\nSkoraj noben graf ni bipartiten.\n\nInt\\(G\\) pada, ko pada tudi radij.\n\nPri n=12 ima bipartitnost ve\u010dji vpil kot globina grafa. \n\n**N =14**\n\nObstaja 509 neizomorfnih kubi\u010dnih grafov na 14 vozli\u0161\u010dih.\n\nInt\\(pot14\\) = 455.\n\nZ ve\u010djim radijem, se lepo ve\u010da tudi indeks.\n\nPrileganje radija in iintervalnega ndeksa je zelo naravno. Ve\u010dji radij, da vi\u0161ji indeks.\n\nNajpogostej\u0161i radij je 4. Odstopanja niso velika.\n\nBipartitni grafi so zelo redki, imajo pa najvi\u0161ji Int\\(G\\).\n\n**N = 16**\n\nObstaja 4060 neizomorfnih kubi\u010dnih grafov na 16 vozli\u0161\u010dih.\n\nInt\\(pot16\\) = \n\nPrileganje radija in iintervalnega ndeksa je zelo naravno. Stopni\u010dasto.", "done": true}︡
︠abc3353f-7846-46f3-87a7-a5bba1c64797︠
m = len(all_cubic)                # number of graphs

rows = []
for i, G in enumerate(all_cubic):
    if i < len(results):
        interval, radius = results[i]
    else:
        interval, radius = "?", "??"

    # per‑graph size; kept constant
    P = G.plot(graph_border=False,
               vertex_size=200)   # adjust once for readability [web:11]

    lbl = text(f"Ind: {i+1}  Int(G): {interval}  Radius: {radius}",
               (0, -0.3), fontsize=10)

    rows.append(P + lbl)

GA = graphics_array(rows, nrows=m, ncols=1)

# overall figure height grows with m
base_height = 3                   # height per graph+label row
GA.save(f"C{n}_all_graphs.pdf",
        axes=False,
        figsize=[8, base_height*m])   # e.g. 5 graphs → height 15 [web:23][web:60]
print(f"File C{n}_all_graphs.pdf generated")
︡89b2c872-7b96-46f4-851f-e8f7e62bd624︡{"stdout": "", "done": true}︡
︠2b5e1d41-5c6c-4b00-b8bd-ff565999a468︠
# Choose a file name; it will appear in the same directory as your worksheet/notebook
import csv  

filename = f"C{n}.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["I","Ind(G)","Radius"])
    # If A is a Sage/Python list of lists or a Sage matrix, this works:
    i=1
    for row in results:
        writer.writerow([i]+list(row))
        i +=1

print("Saved to", filename)
︡377d1ee4-06a9-47e1-98ce-3af398e43a79︡{"stdout": "", "done": true}︡
︠b996b8ec-34ac-4f29-a5ee-511cfc77ce58︠
vert = interval_index(graphs.PathGraph(n))
print(f"Int(PathGraph({n}) = {vert}.")
︡1ad8bda7-edfd-41df-89b7-0042f98f4ec9︡{"html": "<pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">---------------------------------------------------------------------------</span></span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>                                 Traceback (most recent call last)</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[3], line 1</span>\n<span style=\"color: #00aa00\">----&gt; 1</span> vert <span style=\"color: #626262\">=</span> <span style=\"background-color: #aa5500\">interval_index</span>(graphs<span style=\"color: #626262\">.</span>PathGraph(n))\n<span style=\"font-weight: bold; color: #00aa00\">      2</span> <span style=\"color: #008700\">print</span>(<span style=\"color: #af0000\">f</span><span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">Int(PathGraph(</span><span style=\"font-weight: bold; color: #af5f87\">{</span>n<span style=\"font-weight: bold; color: #af5f87\">}</span><span style=\"color: #af0000\">) = </span><span style=\"font-weight: bold; color: #af5f87\">{</span>vert<span style=\"font-weight: bold; color: #af5f87\">}</span><span style=\"color: #af0000\">.</span><span style=\"color: #af0000\">\"</span>)\n</span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>: name 'interval_index' is not defined</span></pre>", "done": true}︡
︠ed2deb7f-bfa6-4f54-b0ec-bf56aeb8fc15︠
print(n)
ind
ind = ind-1
print(results[ind])
all_cubic[ind].plot().show()
︡7ff92992-efa0-46d4-bfcd-2220c29bde57︡{"html": "<pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">---------------------------------------------------------------------------</span></span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>                                 Traceback (most recent call last)</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[6], line 4</span>\n<span style=\"font-weight: bold; color: #00aa00\">      2</span> ind  <span style=\"color: #626262\">=</span> Integer(<span style=\"color: #626262\">19</span>)\n<span style=\"font-weight: bold; color: #00aa00\">      3</span> ind <span style=\"color: #626262\">=</span> ind<span style=\"color: #626262\">-</span>Integer(<span style=\"color: #626262\">1</span>)\n<span style=\"color: #00aa00\">----&gt; 4</span> <span style=\"color: #008700\">print</span>(<span style=\"background-color: #aa5500\">results</span>[ind])\n<span style=\"font-weight: bold; color: #00aa00\">      5</span> all_cubic[ind]<span style=\"color: #626262\">.</span>plot()<span style=\"color: #626262\">.</span>show()\n</span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>: name 'results' is not defined</span></pre>", "done": true}︡
︠29aaa52a-49df-4ef6-bc28-39e058d4e714︠
import csv  
from itertools import combinations

########################################
# 1) Interval-based indices
########################################

def interval_size(G, u, v):
    du = G.distances(source=u)
    dv = G.distances(source=v)
    d_uv = du[v]
    return sum(1 for w in G.vertices() if du[w] + dv[w] == d_uv)

def Int_index(G):
    dist = G.distance_all_pairs()   # dict-of-dicts
    V = G.vertices()
    s = 0
    for u, v in combinations(V, 2):
        d_uv = dist[u][v]
        # count vertices on some shortest u-v path
        cnt = sum(1 for w in V if dist[u][w] + dist[w][v] == d_uv)
        s += cnt - 1
    return s

def wiener_index(G):
    dist = G.distance_all_pairs()
    return sum(dist[u][v] for u, v in combinations(G.vertices(), 2))

########################################
# 2) Graph classification / properties
########################################

def graph_data(G, graph_id):
    IntG = Int_index(G)
    WG   = wiener_index(G)

    return {
        "id": graph_id,
        "n": G.order(),
        "m": G.size(),
        "planar": G.is_planar(),
        "bipartite": G.is_bipartite(),
        "bridgeless": (G.edge_connectivity() >= 2),
        "girth": G.girth(),
        "diameter": G.diameter(),
        "edge_connectivity": G.edge_connectivity(),
        "vertex_connectivity": G.vertex_connectivity(),
        "automorphism_group_order": G.automorphism_group().order(),
        "chromatic_index": G.chromatic_index(),
        "Int(G)": IntG,
        "Wiener(G)": WG,
        "Int(G)-Wiener(G)": IntG - WG
    }

########################################
# 3) Generate cubic graphs
########################################

graphs_n = all_cubic


########################################
# 4) Write CSV
########################################

output_file = f"C{n}_cubic_graphs.csv"

with open(output_file, "w", newline="") as f:
    writer = None

    for i, G in enumerate(graphs_n):
        row = graph_data(G, i)

        if writer is None:
            # first row = column names
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()

        writer.writerow(row)

print(f"CSV written to {output_file}")
︡afb18f14-932a-433d-b682-e1cad9033894︡{"html": "<pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">---------------------------------------------------------------------------</span></span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>                                 Traceback (most recent call last)</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[8], line 59</span>\n<span style=\"font-weight: bold; color: #00aa00\">     37</span>     <span style=\"font-weight: bold; color: #008700\">return</span> {\n<span style=\"font-weight: bold; color: #00aa00\">     38</span>         <span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">id</span><span style=\"color: #af0000\">\"</span>: graph_id,\n<span style=\"font-weight: bold; color: #00aa00\">     39</span>         <span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">n</span><span style=\"color: #af0000\">\"</span>: G<span style=\"color: #626262\">.</span>order(),\n<span style=\"color: #00aa00\">   (...)</span>\n<span style=\"font-weight: bold; color: #00aa00\">     52</span>         <span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">Int(G)-Wiener(G)</span><span style=\"color: #af0000\">\"</span>: IntG <span style=\"color: #626262\">-</span> WG\n<span style=\"font-weight: bold; color: #00aa00\">     53</span>     }\n<span style=\"font-weight: bold; color: #00aa00\">     55</span> <span style=\"font-style: italic; color: #5f8787\">########################################</span>\n<span style=\"font-weight: bold; color: #00aa00\">     56</span> <span style=\"font-style: italic; color: #5f8787\"># 3) Generate cubic graphs</span>\n<span style=\"font-weight: bold; color: #00aa00\">     57</span> <span style=\"font-style: italic; color: #5f8787\">########################################</span>\n<span style=\"color: #00aa00\">---&gt; 59</span> graphs_n <span style=\"color: #626262\">=</span> <span style=\"background-color: #aa5500\">all_cubic</span>\n<span style=\"font-weight: bold; color: #00aa00\">     62</span> <span style=\"font-style: italic; color: #5f8787\">########################################</span>\n<span style=\"font-weight: bold; color: #00aa00\">     63</span> <span style=\"font-style: italic; color: #5f8787\"># 4) Write CSV</span>\n<span style=\"font-weight: bold; color: #00aa00\">     64</span> <span style=\"font-style: italic; color: #5f8787\">########################################</span>\n<span style=\"font-weight: bold; color: #00aa00\">     66</span> output_file <span style=\"color: #626262\">=</span> <span style=\"color: #af0000\">f</span><span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">C</span><span style=\"font-weight: bold; color: #af5f87\">{</span>n<span style=\"font-weight: bold; color: #af5f87\">}</span><span style=\"color: #af0000\">_cubic_graphs.csv</span><span style=\"color: #af0000\">\"</span>\n</span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">NameError</span>: name 'all_cubic' is not defined</span></pre>", "done": true}︡
︠fe530bd8-c638-427c-9ff8-2fa27a320d8c︠
#From these results you can already justify the statement:
#For cubic graphs, the interval index Int(G) is overwhelmingly determined by global distance structure (Wiener index), while classical structural properties (planarity, girth, connectivity, coloring) have negligible explanatory power.
#That is a nontrivial structural insight, and it aligns perfectly with metric graph theory.
#
# However: 
# The key issue: target leakage 
# We are predicting Int(G), and feature list includes:
#Wiener(G)
#Int(G)-Wiener(G)
#But mathematically:
#Int(G)=Wiener(G)+(Int(G)−Wiener(G))
#So the model is being given two parts whose sum is exactly the target.
#Consequence
#This is perfect information leakage.
#The Random Forest is not “discovering” a relationship — it is simply recombining components of provided Int(G) .
#R² ≈ 1
#huge importance for Wiener(G) and Int(G)-Wiener(G)
#everything else being negligible

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def load_xy_from_csv(filename, target_col="Int(G)", drop_cols=("id",)):
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if cols is None:
            raise ValueError("CSV has no header row.")

        drop = set(drop_cols)
        if target_col not in cols:
            raise KeyError(f"Target column '{target_col}' not found. Available: {cols}")

        feature_names = [c for c in cols if (c != target_col and c not in drop)]

        X_rows, y = [], []
        for row in reader:
            y.append(float(row[target_col]))
            feat = []
            for c in feature_names:
                val = row[c]
                if val in ("True", "False"):
                    feat.append(1.0 if val == "True" else 0.0)
                else:
                    feat.append(float(val))
            X_rows.append(feat)

    return np.array(X_rows, dtype=float), np.array(y, dtype=float), feature_names

def rf_analysis(csv_file, target_col="Int(G)", test_size=0.25, random_state=0,
                n_estimators=200, n_repeats=5):

    # Force sklearn-friendly base Python types (important in Sage)
    test_size_py   = float(test_size) if test_size is not None else None
    random_state_py = int(random_state) if random_state is not None else None
    n_estimators_py = int(n_estimators)
    n_repeats_py    = int(n_repeats)

    X, y, feature_names = load_xy_from_csv(csv_file, target_col=target_col, drop_cols=("id",))
    print("File loaded: "+csv_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_py, random_state=random_state_py
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators_py,
        random_state=random_state_py,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    r2_train = rf.score(X_train, y_train)
    r2_test  = rf.score(X_test, y_test)

    imp = rf.feature_importances_

    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats_py,
        random_state=random_state_py,
        n_jobs=1
    )

    perm_mean = perm.importances_mean
    perm_std  = perm.importances_std

    print("=== Random Forest results ===")
    print(f"Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"R^2 train: {r2_train:.4f}")
    print(f"R^2 test : {r2_test:.4f}\n")

    print("Top features (impurity importance):")
    order = np.argsort(imp)[::-1]
    for k in order[:15]:
        print(f"  {feature_names[k]:30s}  {imp[k]:.6f}")

    print("\nTop features (permutation importance on test set):")
    order2 = np.argsort(perm_mean)[::-1]
    for k in order2[:15]:
        print(f"  {feature_names[k]:30s}  {perm_mean[k]:.6f}  +/- {perm_std[k]:.6f}")

    return {
        "rf": rf,
        "feature_names": feature_names,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "impurity_importance": imp,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
    }

csv_file = f"C{n}_cubic_graphs.csv"
print(csv_file)
out = rf_analysis(csv_file, target_col="Int(G)", test_size=0.25, random_state=0)
print(out)
︡dd145a78-3f35-4dd9-a37e-9c5e87789581︡{"html": "<pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">---------------------------------------------------------------------------</span></span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">FileNotFoundError</span>                         Traceback (most recent call last)</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[5], line 118</span>\n<span style=\"font-weight: bold; color: #00aa00\">    116</span> csv_file <span style=\"color: #626262\">=</span> <span style=\"color: #af0000\">f</span><span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">C</span><span style=\"font-weight: bold; color: #af5f87\">{</span>n<span style=\"font-weight: bold; color: #af5f87\">}</span><span style=\"color: #af0000\">_cubic_graphs.csv</span><span style=\"color: #af0000\">\"</span>\n<span style=\"font-weight: bold; color: #00aa00\">    117</span> <span style=\"color: #008700\">print</span>(csv_file)\n<span style=\"color: #00aa00\">--&gt; 118</span> out <span style=\"color: #626262\">=</span> <span style=\"background-color: #aa5500\">rf_analysis</span><span style=\"background-color: #aa5500\">(</span><span style=\"background-color: #aa5500\">csv_file</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">target_col</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"color: #af0000; background-color: #aa5500\">Int(G)</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">test_size</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"background-color: #aa5500\">RealNumber</span><span style=\"background-color: #aa5500\">(</span><span style=\"color: #af0000; background-color: #aa5500\">'</span><span style=\"color: #af0000; background-color: #aa5500\">0.25</span><span style=\"color: #af0000; background-color: #aa5500\">'</span><span style=\"background-color: #aa5500\">)</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">random_state</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"background-color: #aa5500\">Integer</span><span style=\"background-color: #aa5500\">(</span><span style=\"color: #626262; background-color: #aa5500\">0</span><span style=\"background-color: #aa5500\">)</span><span style=\"background-color: #aa5500\">)</span>\n<span style=\"font-weight: bold; color: #00aa00\">    119</span> <span style=\"color: #008700\">print</span>(out)\n</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[5], line 62</span>, in <span style=\"color: #00aaaa\">rf_analysis</span><span style=\"color: #0000aa\">(csv_file, target_col, test_size, random_state, n_estimators, n_repeats)</span>\n<span style=\"font-weight: bold; color: #00aa00\">     59</span> n_estimators_py <span style=\"color: #626262\">=</span> <span style=\"color: #008700\">int</span>(n_estimators)\n<span style=\"font-weight: bold; color: #00aa00\">     60</span> n_repeats_py    <span style=\"color: #626262\">=</span> <span style=\"color: #008700\">int</span>(n_repeats)\n<span style=\"color: #00aa00\">---&gt; 62</span> X, y, feature_names <span style=\"color: #626262\">=</span> <span style=\"background-color: #aa5500\">load_xy_from_csv</span><span style=\"background-color: #aa5500\">(</span><span style=\"background-color: #aa5500\">csv_file</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">target_col</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"background-color: #aa5500\">target_col</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">drop_cols</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"background-color: #aa5500\">(</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"color: #af0000; background-color: #aa5500\">id</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\">)</span><span style=\"background-color: #aa5500\">)</span>\n<span style=\"font-weight: bold; color: #00aa00\">     63</span> <span style=\"color: #008700\">print</span>(<span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">File loaded: </span><span style=\"color: #af0000\">\"</span><span style=\"color: #626262\">+</span>csv_file)\n<span style=\"font-weight: bold; color: #00aa00\">     65</span> X_train, X_test, y_train, y_test <span style=\"color: #626262\">=</span> train_test_split(\n<span style=\"font-weight: bold; color: #00aa00\">     66</span>     X, y, test_size<span style=\"color: #626262\">=</span>test_size_py, random_state<span style=\"color: #626262\">=</span>random_state_py\n<span style=\"font-weight: bold; color: #00aa00\">     67</span> )\n</span></pre><br/><pre><span style=\"font-family:monospace;\">Cell <span style=\"color: #00aa00\">In[5], line 27</span>, in <span style=\"color: #00aaaa\">load_xy_from_csv</span><span style=\"color: #0000aa\">(filename, target_col, drop_cols)</span>\n<span style=\"font-weight: bold; color: #00aa00\">     26</span> <span style=\"font-weight: bold; color: #008700\">def</span><span style=\"color: #bcbcbc\"> </span><span style=\"color: #0000ff\">load_xy_from_csv</span>(filename, target_col<span style=\"color: #626262\">=</span><span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">Int(G)</span><span style=\"color: #af0000\">\"</span>, drop_cols<span style=\"color: #626262\">=</span>(<span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">id</span><span style=\"color: #af0000\">\"</span>,)):\n<span style=\"color: #00aa00\">---&gt; 27</span>     <span style=\"font-weight: bold; color: #008700\">with</span> <span style=\"color: #008700; background-color: #aa5500\">open</span><span style=\"background-color: #aa5500\">(</span><span style=\"background-color: #aa5500\">filename</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"background-color: #aa5500\">newline</span><span style=\"color: #626262; background-color: #aa5500\">=</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"color: #af0000; background-color: #aa5500\">\"</span><span style=\"background-color: #aa5500\">)</span> <span style=\"font-weight: bold; color: #008700\">as</span> f:\n<span style=\"font-weight: bold; color: #00aa00\">     28</span>         reader <span style=\"color: #626262\">=</span> csv<span style=\"color: #626262\">.</span>DictReader(f)\n<span style=\"font-weight: bold; color: #00aa00\">     29</span>         cols <span style=\"color: #626262\">=</span> reader<span style=\"color: #626262\">.</span>fieldnames\n</span></pre><br/><pre><span style=\"font-family:monospace;\">File <span style=\"color: #00aa00\">/ext/sage/10.7/local/var/lib/sage/venv-python3.12.5/lib/python3.12/site-packages/IPython/core/interactiveshell.py:310</span>, in <span style=\"color: #00aaaa\">_modified_open</span><span style=\"color: #0000aa\">(file, *args, **kwargs)</span>\n<span style=\"font-weight: bold; color: #00aa00\">    303</span> <span style=\"font-weight: bold; color: #008700\">if</span> file <span style=\"font-weight: bold; color: #af00ff\">in</span> {<span style=\"color: #626262\">0</span>, <span style=\"color: #626262\">1</span>, <span style=\"color: #626262\">2</span>}:\n<span style=\"font-weight: bold; color: #00aa00\">    304</span>     <span style=\"font-weight: bold; color: #008700\">raise</span> <span style=\"font-weight: bold; color: #d75f5f\">ValueError</span>(\n<span style=\"font-weight: bold; color: #00aa00\">    305</span>         <span style=\"color: #af0000\">f</span><span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">IPython won</span><span style=\"color: #af0000\">'</span><span style=\"color: #af0000\">t let you open fd=</span><span style=\"font-weight: bold; color: #af5f87\">{</span>file<span style=\"font-weight: bold; color: #af5f87\">}</span><span style=\"color: #af0000\"> by default </span><span style=\"color: #af0000\">\"</span>\n<span style=\"font-weight: bold; color: #00aa00\">    306</span>         <span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">as it is likely to crash IPython. If you know what you are doing, </span><span style=\"color: #af0000\">\"</span>\n<span style=\"font-weight: bold; color: #00aa00\">    307</span>         <span style=\"color: #af0000\">\"</span><span style=\"color: #af0000\">you can use builtins</span><span style=\"color: #af0000\">'</span><span style=\"color: #af0000\"> open.</span><span style=\"color: #af0000\">\"</span>\n<span style=\"font-weight: bold; color: #00aa00\">    308</span>     )\n<span style=\"color: #00aa00\">--&gt; 310</span> <span style=\"font-weight: bold; color: #008700\">return</span> <span style=\"background-color: #aa5500\">io_open</span><span style=\"background-color: #aa5500\">(</span><span style=\"background-color: #aa5500\">file</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"color: #626262; background-color: #aa5500\">*</span><span style=\"background-color: #aa5500\">args</span><span style=\"background-color: #aa5500\">,</span><span style=\"background-color: #aa5500\"> </span><span style=\"color: #626262; background-color: #aa5500\">*</span><span style=\"color: #626262; background-color: #aa5500\">*</span><span style=\"background-color: #aa5500\">kwargs</span><span style=\"background-color: #aa5500\">)</span>\n</span></pre><br/><pre><span style=\"font-family:monospace;\"><span style=\"color: #aa0000\">FileNotFoundError</span>: [Errno 2] No such file or directory: 'C&lt;function numerical_approx at 0x7f5c5e2c4900&gt;_cubic_graphs.csv'</span></pre>", "done": true}︡
︠de7809b7-129c-4082-89e4-aaa356b807ff︠
# Leakage-free Random Forest analysis for Int(G) (Sage-friendly)
# - removes "Wiener(G)" and "Int(G)-Wiener(G)" from features by default
# - also supports predicting the residual Int(G)-Wiener(G) as a separate target

import os
# (recommended) limit native threading to avoid kernel crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def load_xy_from_csv(
    filename,
    target_col="Int(G)",
    drop_cols=("id",),
    drop_feature_cols=(),
):
    """
    Load CSV -> (X, y, feature_names)

    - target_col: column to predict
    - drop_cols: columns never used (e.g. id)
    - drop_feature_cols: columns explicitly excluded from FEATURES (e.g. leakage)
    """
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if cols is None:
            raise ValueError("CSV has no header row.")
        if target_col not in cols:
            raise KeyError(f"Target column '{target_col}' not found. Available: {cols}")

        drop_cols = set(drop_cols)
        drop_feature_cols = set(drop_feature_cols)

        feature_names = [
            c for c in cols
            if (c != target_col) and (c not in drop_cols) and (c not in drop_feature_cols)
        ]

        X_rows, y = [], []
        for row in reader:
            # target
            y.append(float(row[target_col]))

            # features
            feat = []
            for c in feature_names:
                val = row[c]
                if val in ("True", "False"):
                    feat.append(1.0 if val == "True" else 0.0)
                else:
                    # handle Sage "Infinity" if it ever appears
                    if val in ("+Infinity", "Infinity", "inf", "Inf"):
                        feat.append(float("inf"))
                    else:
                        feat.append(float(val))
            X_rows.append(feat)

    X = np.array(X_rows, dtype=float)
    y = np.array(y, dtype=float)

    # If any inf values exist, you must decide how to handle them.
    # For cubic connected graphs, girth/diameter/etc should be finite.
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values (inf/nan) found in features. Clean your CSV or drop those columns.")

    return X, y, feature_names


def rf_analysis_no_leakage(
    csv_file,
    target_col="Int(G)",
    leakage_cols=("Wiener(G)", "Int(G)-Wiener(G)"),
    test_size=0.25,
    random_state=0,
    n_estimators=300,
    n_repeats=10,
):
    """
    Random Forest regression with leakage columns removed from features.

    Returns a dict with model + importances.
    """
    # Force sklearn-friendly base Python types (important in Sage)
    test_size_py = float(test_size) if test_size is not None else None
    rs_py = int(random_state) if random_state is not None else None

    X, y, feature_names = load_xy_from_csv(
        csv_file,
        target_col=target_col,
        drop_cols=("id",),
        drop_feature_cols=leakage_cols,   # <-- leakage removed here
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_py, random_state=rs_py
    )

    # n_jobs=1 is safest in Sage/Jupyter
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=rs_py,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)

    r2_train = rf.score(X_train, y_train)
    r2_test  = rf.score(X_test, y_test)

    # Permutation importance (most meaningful)
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=int(n_repeats),
        random_state=rs_py,
        n_jobs=1
    )
    perm_mean = perm.importances_mean
    perm_std  = perm.importances_std

    # Impurity-based importance (can be biased, still useful)
    imp = rf.feature_importances_

    print("=== Random Forest (LEAKAGE REMOVED) ===")
    print(f"File: {csv_file}")
    print(f"Target: {target_col}")
    print(f"Removed leakage cols from features: {list(leakage_cols)}")
    print(f"Samples: {len(y)} | Features used: {len(feature_names)}")
    print(f"R^2 train: {r2_train:.4f}")
    print(f"R^2 test : {r2_test:.4f}\n")

    order_imp = np.argsort(imp)[::-1]
    print("Top features (impurity importance):")
    for k in order_imp[:15]:
        print(f"  {feature_names[k]:30s}  {imp[k]:.6f}")

    order_perm = np.argsort(perm_mean)[::-1]
    print("\nTop features (permutation importance on test set):")
    for k in order_perm[:15]:
        print(f"  {feature_names[k]:30s}  {perm_mean[k]:.6f}  +/- {perm_std[k]:.6f}")

    return {
        "rf": rf,
        "feature_names": feature_names,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "impurity_importance": imp,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
    }


# -----------------------------
# USAGE EXAMPLES
# -----------------------------

n = 18
csv_file = f"C{n}_cubic_graphs.csv"

# A) Predict Int(G) using ONLY non-leakage properties
out_int = rf_analysis_no_leakage(
    csv_file,
    target_col="Int(G)",
    leakage_cols=("Wiener(G)", "Int(G)-Wiener(G)"),
    test_size=0.25,
    random_state=0,
    n_estimators=300,
    n_repeats=10
)

# B) (Recommended) Predict the "extra geodesic mass" Int(G)-Wiener(G)
#    using classical invariants (this is often more interesting scientifically)
out_extra = rf_analysis_no_leakage(
    csv_file,
    target_col="Int(G)-Wiener(G)",
    leakage_cols=("Int(G)", "Wiener(G)"),  # remove parts that trivially reconstruct the target
    test_size=0.25,
    random_state=0,
    n_estimators=300,
    n_repeats=10
)
︡6db1d888-0deb-4ed6-a887-0df4d8d6f61d︡{"html": "<pre><span style=\"font-family:monospace;\">=== Random Forest (LEAKAGE REMOVED) ===\nFile: C18_cubic_graphs.csv\nTarget: Int(G)\nRemoved leakage cols from features: ['Wiener(G)', 'Int(G)-Wiener(G)']\nSamples: 33561 | Features used: 11\nR^2 train: 0.6405\nR^2 test : 0.6268\n\nTop features (impurity importance):\n  diameter                        0.842418\n  bipartite                       0.075035\n  automorphism_group_order        0.059547\n  girth                           0.007789\n  planar                          0.006614\n  edge_connectivity               0.002599\n  chromatic_index                 0.002549\n  vertex_connectivity             0.002227\n  bridgeless                      0.001223\n  m                               0.000000\n  n                               0.000000\n\nTop features (permutation importance on test set):\n  diameter                        0.925363  +/- 0.015761\n  automorphism_group_order        0.104186  +/- 0.004294\n  bipartite                       0.062676  +/- 0.002487\n  girth                           0.010594  +/- 0.001815\n  planar                          0.007585  +/- 0.001249\n  edge_connectivity               0.004350  +/- 0.001031\n  vertex_connectivity             0.003521  +/- 0.000920\n  chromatic_index                 0.003241  +/- 0.000660\n  bridgeless                      0.000083  +/- 0.000139\n  m                               0.000000  +/- 0.000000\n  n                               0.000000  +/- 0.000000\n</span></pre><br/><pre><span style=\"font-family:monospace;\">=== Random Forest (LEAKAGE REMOVED) ===\nFile: C18_cubic_graphs.csv\nTarget: Int(G)-Wiener(G)\nRemoved leakage cols from features: ['Int(G)', 'Wiener(G)']\nSamples: 33561 | Features used: 11\nR^2 train: 0.2331\nR^2 test : 0.1797\n\nTop features (impurity importance):\n  bipartite                       0.512189\n  diameter                        0.230926\n  automorphism_group_order        0.095442\n  girth                           0.070283\n  edge_connectivity               0.023020\n  planar                          0.020080\n  vertex_connectivity             0.019547\n  chromatic_index                 0.016333\n  bridgeless                      0.012180\n  m                               0.000000\n  n                               0.000000\n\nTop features (permutation importance on test set):\n  diameter                        0.359495  +/- 0.007545\n  bipartite                       0.167692  +/- 0.006421\n  automorphism_group_order        0.106037  +/- 0.007220\n  girth                           0.056341  +/- 0.004891\n  edge_connectivity               0.042074  +/- 0.002617\n  vertex_connectivity             0.030553  +/- 0.002165\n  bridgeless                      0.011901  +/- 0.001395\n  chromatic_index                 0.011732  +/- 0.001626\n  planar                          0.008095  +/- 0.001703\n  m                               0.000000  +/- 0.000000\n  n                               0.000000  +/- 0.000000\n</span></pre>", "done": true}︡
︠83ff0cfe-2bb1-4ae7-b0a2-6e920460aa59︠
import sklearn
print(sklearn.__version__)

from sklearn import datasets
iris = datasets.load_iris()
print(iris.data.shape)
︡d453335e-7a8a-4a84-8c27-d991f749214b︡{"html": "<pre><span style=\"font-family:monospace;\">1.8.0\n(150, 4)\n</span></pre>", "done": true}︡
︠c9e81cf5-63d0-44d4-8614-c01afa10da14︠
# Combined (multi-n) leakage-free analysis for Int(G) from CSVs:
#   files: C{n}_cubic_graphs.csv for even n in [nMin, nMax]
# Runs RandomForest with:
#   - leakage columns removed from FEATURES
#   - optional normalization of target to reduce size-dominance
#   - option to test on an unseen n (recommended)

import os
# Prevent kernel crashes from OpenMP/BLAS oversubscription (set BEFORE numpy/sklearn import)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import math
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# -----------------------------
# User settings
# -----------------------------
nMin = 6
nMax = 18

# Choose ONE target mode:
TARGET_COL = "Int(G)"          # raw target from CSV
NORMALIZE_TARGET = True        # if True: predict Int(G)/C(n,2) instead of Int(G)

# Leakage removal (for predicting Int(G) fairly)
LEAKAGE_COLS = ("Wiener(G)", "Int(G)-Wiener(G)")

# Split mode:
TEST_ON_UNSEEN_N = True        # recommended: hold out one n completely
HELD_OUT_N = nMax              # which n to hold out if TEST_ON_UNSEEN_N=True

# RF parameters (safe in notebooks)
N_ESTIMATORS = 300
N_REPEATS_PI = 10
RANDOM_STATE = 0


# -----------------------------
# Helpers
# -----------------------------
def choose2(n: int) -> int:
    return n * (n - 1) // 2

def file_for_n(n: int) -> str:
    return f"C{n}_cubic_graphs.csv"

def read_rows_with_n(filename: str, n_value: int):
    """Read CSV rows and add an explicit 'n_from_filename'."""
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{filename}: missing header row")
        rows = []
        for r in reader:
            r["n_from_filename"] = str(n_value)
            rows.append(r)
        return reader.fieldnames + ["n_from_filename"], rows

def load_combined_data(nMin: int, nMax: int):
    """
    Loads all available C{n}_cubic_graphs.csv (even n).
    Returns (all_rows, all_columns).
    """
    all_rows = []
    all_cols = None

    for n in range(nMin, nMax + 1, 2):
        fn = file_for_n(n)
        if not os.path.exists(fn):
            print(f"[skip] missing file: {fn}")
            continue

        cols, rows = read_rows_with_n(fn, n)
        print(f"[load] {fn}: {len(rows)} rows")

        # Ensure consistent columns across files
        if all_cols is None:
            all_cols = cols
        else:
            # Require same base columns (order can differ; DictReader uses names)
            missing = set(all_cols) - set(cols)
            extra = set(cols) - set(all_cols)
            if missing or extra:
                raise ValueError(
                    f"Column mismatch in {fn}.\nMissing: {missing}\nExtra: {extra}"
                )

        all_rows.extend(rows)

    if all_cols is None:
        raise ValueError("No input files were found/loaded.")

    return all_rows, all_cols

def build_Xy(rows, target_col, leakage_cols, normalize_target=True):
    """
    Build X,y from dict rows:
      - target is Int(G) (raw) or Int(G)/C(n,2) if normalize_target
      - features exclude leakage cols and 'id' (position is index)
      - we include 'n_from_filename' as a feature (important for combined analysis)
    """
    drop_cols = {"id"}  # drop id if present
    leakage_cols = set(leakage_cols)

    # Determine available columns from first row
    cols = list(rows[0].keys())
    if target_col not in cols:
        raise KeyError(f"Target column '{target_col}' not found. Available: {cols}")

    # Feature set
    feature_names = []
    for c in cols:
        if c == target_col:
            continue
        if c in drop_cols:
            continue
        if c in leakage_cols:
            continue
        # Always keep n_from_filename
        feature_names.append(c)

    X = []
    y = []

    for r in rows:
        n_val = int(r["n_from_filename"])
        # target
        t = float(r[target_col])
        if normalize_target:
            t = t / float(choose2(n_val))
        y.append(t)

        # features
        feat = []
        for c in feature_names:
            v = r[c]
            if v in ("True", "False"):
                feat.append(1.0 if v == "True" else 0.0)
            else:
                if v in ("+Infinity", "Infinity", "inf", "Inf", "nan", "NaN", ""):
                    raise ValueError(f"Non-finite value in column '{c}': {v}")
                feat.append(float(v))
        X.append(feat)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y, feature_names

def print_top_importances(feature_names, values, std=None, topk=15, title=""):
    order = np.argsort(values)[::-1]
    if title:
        print(title)
    for k in order[:topk]:
        if std is None:
            print(f"  {feature_names[k]:30s} {values[k]:.6f}")
        else:
            print(f"  {feature_names[k]:30s} {values[k]:.6f} +/- {std[k]:.6f}")


# -----------------------------
# Main: load, split, train, report
# -----------------------------
rows, cols = load_combined_data(nMin, nMax)
X, y, feature_names = build_Xy(
    rows,
    target_col=TARGET_COL,
    leakage_cols=LEAKAGE_COLS,
    normalize_target=NORMALIZE_TARGET
)

# Choose split
rs = int(RANDOM_STATE)

if TEST_ON_UNSEEN_N:
    # Hold out all graphs with n_from_filename == HELD_OUT_N
    n_feat_idx = feature_names.index("n_from_filename")
    n_values = X[:, n_feat_idx].astype(int)

    test_mask = (n_values == int(HELD_OUT_N))
    train_mask = ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test   = X[test_mask],  y[test_mask]

    print(f"\n=== Split: train on n in [{nMin},{nMax}]\\{{{HELD_OUT_N}}}, test on n={HELD_OUT_N} ===")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=rs
    )
    print("\n=== Split: random 75/25 ===")

print(f"Samples total: {len(y)} | train: {len(y_train)} | test: {len(y_test)}")
print(f"Features used: {len(feature_names)}")
print(f"Target: {TARGET_COL} {'(normalized by C(n,2))' if NORMALIZE_TARGET else '(raw)'}")
print(f"Leakage removed from features: {list(LEAKAGE_COLS)}\n")

# Train RF (safe settings: n_jobs=1)
rf = RandomForestRegressor(
    n_estimators=int(N_ESTIMATORS),
    random_state=rs,
    n_jobs=1
)
rf.fit(X_train, y_train)

r2_train = rf.score(X_train, y_train)
r2_test  = rf.score(X_test, y_test)

print("=== Random Forest (combined n, leakage removed) ===")
print(f"R^2 train: {r2_train:.4f}")
print(f"R^2 test : {r2_test:.4f}\n")

# Importances
imp = rf.feature_importances_
print_top_importances(feature_names, imp, title="Top features (impurity importance):")

perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=int(N_REPEATS_PI),
    random_state=rs,
    n_jobs=1
)
print()
print_top_importances(feature_names, perm.importances_mean, perm.importances_std,
                      title="Top features (permutation importance on test set):")

# Optional: quick sanity check about the n feature
if "n_from_filename" in feature_names:
    i = feature_names.index("n_from_filename")
    print(f"\nPermutation importance of n_from_filename: {perm.importances_mean[i]:.6f} +/- {perm.importances_std[i]:.6f}")
︡d3d0fba4-bdd8-4be3-9f2a-05aa205ab5e8︡{"html": "<pre><span style=\"font-family:monospace;\">[load] C6_cubic_graphs.csv: 2 rows\n[load] C8_cubic_graphs.csv: 5 rows\n[load] C10_cubic_graphs.csv: 19 rows\n[load] C12_cubic_graphs.csv: 85 rows\n[load] C14_cubic_graphs.csv: 509 rows\n[skip] missing file: C16_cubic_graphs.csv\n</span></pre><br/><pre><span style=\"font-family:monospace;\">[load] C18_cubic_graphs.csv: 33561 rows\n</span></pre><br/><pre><span style=\"font-family:monospace;\">\n=== Split: train on n in [6,18]\\{18}, test on n=18 ===\nSamples total: 34181 | train: 620 | test: 33561\nFeatures used: 12\nTarget: Int(G) (normalized by C(n,2))\nLeakage removed from features: ['Wiener(G)', 'Int(G)-Wiener(G)']\n\n</span></pre><br/><pre><span style=\"font-family:monospace;\">=== Random Forest (combined n, leakage removed) ===\nR^2 train: 0.7909\nR^2 test : -0.8123\n\nTop features (impurity importance):\n  diameter                       0.662466\n  bipartite                      0.152152\n  automorphism_group_order       0.088702\n  n_from_filename                0.015555\n  m                              0.015499\n  n                              0.014964\n  girth                          0.014547\n  planar                         0.014361\n  edge_connectivity              0.007928\n  vertex_connectivity            0.007768\n  chromatic_index                0.004302\n  bridgeless                     0.001757\n</span></pre><br/><pre><span style=\"font-family:monospace;\">\nTop features (permutation importance on test set):\n  diameter                       0.993061 +/- 0.007135\n  bipartite                      0.040436 +/- 0.000780\n  chromatic_index                0.003520 +/- 0.000249\n  bridgeless                     0.001274 +/- 0.000134\n  n                              0.000000 +/- 0.000000\n  m                              0.000000 +/- 0.000000\n  n_from_filename                0.000000 +/- 0.000000\n  automorphism_group_order       -0.029744 +/- 0.001759\n  planar                         -0.032315 +/- 0.000742\n  vertex_connectivity            -0.033202 +/- 0.000486\n  edge_connectivity              -0.035491 +/- 0.000510\n  girth                          -0.051656 +/- 0.000987\n\nPermutation importance of n_from_filename: 0.000000 +/- 0.000000\n</span></pre>", "done": true}︡
