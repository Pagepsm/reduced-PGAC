# 数式で説明する研究方法と実験設計

本研究では、状態数100の未知線形システムを用い、**どの状態方向を学習に残すか**が制御性能・モデル推定条件・計算時間に与える影響を調べた。初期データの収集、SVDによる縮約、オンラインのモデル推定と方策更新、元の100次元系での評価という順に実験を組み立てている。

## 1. 制御するシステム

対象は離散時間線形時不変系である。

```math
x_{t+1}=Ax_t+Bu_t+w_t,
\qquad x_t\in\mathbb{R}^{n},\quad u_t\in\mathbb{R}^{m}.
```

$`x_t`$ は状態、$`u_t`$ は制御入力、$`w_t`$ は平均零・共分散 $`W`$ のプロセスノイズである。行列 $`A\in\mathbb{R}^{n\times n}`$、$`B\in\mathbb{R}^{n\times m}`$ は制御器にとって未知であり、シミュレータ内で状態更新と評価を行うときに用いる。

今回の実験では $`n=100`$、$`m=4`$ とした。プラントは安定な合成系で、初期の安定化ゲインとして $`K_0=0`$ を用いる。状態・入力の時系列からモデルを推定し、フィードバックゲインを更新してLQRコストを下げる。

## 2. 100次元合成プラントの構成

### 2.1 24方向を支配的にする狙い

状態空間をランダムな直交基底で混合したうえで、各モードの励起の強さと制御上の重要度を分けて設計した。100方向の内訳は次のとおりである。

| 群 | モードの番号 | 次元 | 役割 |
|---|---|---:|---|
| Strong | $`1,\ldots,12`$ | 12 | 初期データで大きく変動する |
| Important | $`13,\ldots,24`$ | 12 | 変動はやや小さいが、制御コストで重視する |
| Weak | $`25,\ldots,100`$ | 76 | 入力・初期値・外乱の影響を小さく設定する |

設計した支配的な状態方向は $`12+12=24`$ 次元である。**24次元付近を制御性能とモデル推定のトレードオフが現れやすい境界として設計した。** SVDへの入力は実際に収集した状態データであり、射影行列へ24方向を直接指定する方法は採っていない。

### 2.2 直交基底とシステム行列

標準正規乱数から生成した行列をQR分解し、直交行列 $`V\in\mathbb{R}^{100\times100}`$ を構成する。

```math
V^\top V=VV^\top=I_{100},\qquad z_t=V^\top x_t.
```

$`z_t`$ がモード座標である。モード空間で対角な状態行列 $`A_{\mathrm m}`$ を作り、物理座標へ戻す。

```math
A_{\mathrm m}=\mathrm{diag}(\lambda_1,\ldots,\lambda_{100}),
\qquad A=VA_{\mathrm m}V^\top.
```

固有値は各群の区間内で等間隔に配置した。

| 群 | 固有値 $`\lambda_i`$ の区間 | モード数 |
|---|---:|---:|
| Strong | $`[0.940,0.975]`$ | 12 |
| Important | $`[0.980,0.997]`$ | 12 |
| Weak | $`[0.450,0.880]`$ | 76 |

すべての固有値が単位円内にあり、$`\rho(A)<1`$ を満たす。Important群は入力による励起がStrong群より弱く、固有値が1に近いため状態の減衰に時間がかかる。

### 2.3 入力を全モードに与える

$`G\in\mathbb{R}^{100\times4}`$ の各要素を標準正規乱数から生成し、モードごとに入力スケール $`s_i`$ を掛ける。

```math
B_{\mathrm m}=D_BG,\qquad
D_B=\mathrm{diag}(s_1,\ldots,s_{100}),\qquad B=VB_{\mathrm m}.
```

```math
s_i=\begin{cases}
0.12 & (1\le i\le12),\\
0.07 & (13\le i\le24),\\
2\times10^{-4} & (25\le i\le100).
\end{cases}
```

Weak群の入力スケールはStrong群の600分の1、Important群の350分の1である。これにより、入力で励起される状態変動の大きさを群ごとに分けた。**Weak群も入力係数をゼロにはせず、可制御性はPBH判定で数値的に確認した。**

### 2.4 状態重み、外乱、初期状態

LQRの状態重みもモードごとに設定した。物理座標の重み行列は次式である。

```math
Q_{\mathrm m}=\mathrm{diag}(q_1,\ldots,q_{100}),
\qquad Q=VQ_{\mathrm m}V^\top,\qquad R=0.1I_4.
```

外乱と初期状態はモード座標で生成する。

```math
w_t=Vw_t^{\mathrm m},\qquad
w_t^{\mathrm m}\sim\mathcal N(0,W_{\mathrm m}),
\qquad W=VW_{\mathrm m}V^\top.
```

```math
W_{\mathrm m}=\mathrm{diag}(\sigma_{w,1}^{2},\ldots,\sigma_{w,100}^{2}),
\qquad z_0\sim\mathcal N(0,\Sigma_{0,\mathrm m}),\qquad x_0=Vz_0.
```

ここで、$`\Sigma_{0,\mathrm m}=\mathrm{diag}(\sigma_{0,1}^{2},\ldots,\sigma_{0,100}^{2})`$ である。具体値をまとめる。

| 設定 | Strong（12） | Important（12） | Weak（76） |
|---|---:|---:|---:|
| 入力スケール $`s_i`$ | 0.12 | 0.07 | $`2\times10^{-4}`$ |
| 状態重み $`q_i`$ | 1 | 15 | 0.05 |
| プロセスノイズの標準偏差 $`\sigma_{w,i}`$ | 0.012 | 0.003 | $`10^{-6}`$ |
| 初期状態の標準偏差 $`\sigma_{0,i}`$ | 0.2 | 0.03 | $`10^{-5}`$ |

Important群を $`q_i=15`$ としたのは、12次元まで縮約したときに落とす方向が、LQRコスト上で意味を持つようにするためである。Weak群の状態変動は入力・初期値・外乱のいずれからも小さくし、データの特異値構造に24方向の境界を作った。

### 2.5 初期データの収集とSVDに現れる構造

初期段階では $`t_0=600`$ ステップの状態データを集める。探索入力はRademacher型の符号列とし、各入力成分は独立に正負の値を取る。

```math
u_t=K_0x_t+e_t=e_t,\qquad
e_t=0.05\,\mathrm{sign}(g_t),\qquad g_t\sim\mathcal N(0,I_4).
```

したがって、$`\mathbb E[e_te_t^\top]=0.05^2I_4`$ である。対角モードの第 $`i`$ 成分は

```math
z_{i,t+1}=\lambda_i z_{i,t}+b_i^\top e_t+w_{i,t}^{\mathrm m}
```

と書ける。$`b_i^\top`$ は $`B_{\mathrm m}`$ の第 $`i`$ 行である。独立な零平均の初期値・探索入力・外乱を用いると、状態の二乗平均は

```math
\mathbb E[z_{i,t}^2]=\lambda_i^{2t}\sigma_{0,i}^2+
\sum_{k=0}^{t-1}\lambda_i^{2k}
\left(0.05^2\|b_i\|_2^2+\sigma_{w,i}^2\right).
```

安定モードの定常状態では、

```math
\lim_{t\to\infty}\mathbb E[z_{i,t}^2]
=\frac{0.05^2\|b_i\|_2^2+\sigma_{w,i}^2}{1-\lambda_i^2}.
```

入力係数、外乱、初期状態、固有値の配置を使い、Strong群に大きい状態エネルギー、Important群に比較的長く残る状態変動、Weak群に小さい状態変動を作った。**24次元の射影が選ばれやすい理由は、投影先を固定したからではなく、このデータ生成構造にある。**

### 2.6 設計過程で検討した不安定プラント

安定な3群モデルとは別に、初期ゲインで安定化する開ループ不安定系も構成した。まずモード空間に安定な基準閉ループ行列 $`A_{\mathrm{cl},0}^{\mathrm m}`$ を作る。対角モードへ小さなブロック間結合を加えた後、

```math
K_0^{\mathrm m}=-\alpha B_{\mathrm m}^\top,\qquad
A_{\mathrm m}=A_{\mathrm{cl},0}^{\mathrm m}+\alpha B_{\mathrm m}B_{\mathrm m}^\top
```

とする。このとき

```math
A_{\mathrm m}+B_{\mathrm m}K_0^{\mathrm m}=A_{\mathrm{cl},0}^{\mathrm m}
```

が成り立つ。$`\alpha`$ を二分探索で調整し、開ループのスペクトル半径を約 $`1.0015`$ にした。入力の強い24方向に不安定性を寄せながら、初期ゲインで安定化できる構成である。

その後、Weak群の入力係数を $`5\times10^{-5}`$、外乱標準偏差を $`10^{-8}`$、初期値標準偏差を $`10^{-6}`$ とした強化版も検討した。Important群の状態重みを20、初期データ長を450に変更し、有限データでの次元選択の影響を観察する設計である。**本ページに掲載する代表的な数値結果は、初期データ長600の比較実験である。**

## 3. 制御目標：LQRコスト

固定ゲイン $`K\in\mathbb R^{m\times n}`$ に対する状態フィードバック $`u_t=Kx_t`$ を考える。閉ループ行列を

```math
A_K=A+BK
```

とする。$`\rho(A_K)<1`$ のとき、状態共分散 $`\Sigma_K`$ はLyapunov方程式

```math
\Sigma_K=W+A_K\Sigma_KA_K^\top
```

を満たす。LQRコストは

```math
C(K)=\mathrm{Tr}\!\left((Q+K^\top RK)\Sigma_K\right)
```

である。$`Q\succ0`$ と $`R\succ0`$ は状態・入力の重みを表す。$`W\succeq0`$ についてもコストを定義し、真の100次元系の最適ゲイン $`K^\star`$ を比較基準に用いる。制御器の学習に用いるのは状態・入力の観測データであり、真の $`A,B`$ を使ったLQR評価はシミュレーション上で別に行う。

## 4. 全状態の間接型PGAC

探索入力 $`e_t`$ を含めて

```math
u_t=K_tx_t+e_t
```

を印加する。回帰ベクトルと推定モデルを

```math
\phi_t=\left[u_t^\top\;x_t^\top\right]^\top\in\mathbb R^{m+n},
\qquad \widehat\Theta_t=\left[\widehat B_t\quad\widehat A_t\right]
```

と置く。RLSで $`\widehat\Theta_t`$ を更新し、推定モデル上でLQR方策勾配を計算する。

真のモデルが既知の場合の勾配は、$`S_K`$ を次のLyapunov方程式の解として

```math
S_K=Q+K^\top RK+A_K^\top S_KA_K
```

```math
\nabla C(K)=2\left((R+B^\top S_KB)K+B^\top S_KA\right)\Sigma_K
```

である。間接型PGACでは推定モデルを代入し、$`K_{t+1}=K_t-\eta\widehat{\nabla C}_{t+1}(K_t)`$ と更新する。$`\eta`$ はステップサイズである。

## 5. 提案手法：初期データのSVDと低次元PGAC

### 5.1 射影の構成

初期状態データ行列を

```math
X_{0,t_0}=\left[x_0\;x_1\;\cdots\;x_{t_0-1}\right]\in\mathbb R^{n\times t_0}
```

とし、SVDを

```math
X_{0,t_0}=U\Lambda V_X^\top
```

と書く。$`U_{n_r}`$ を上位 $`n_r`$ 本の左特異ベクトルとすると、

```math
P=U_{n_r}^\top\in\mathbb R^{n_r\times n},\qquad
PP^\top=I_{n_r},\qquad \xi_t=Px_t.
```

縮約次元24の場合は $`P\in\mathbb R^{24\times100}`$ である。初期データから一度決めた $`P`$ をオンライン段階で固定する。

### 5.2 縮約状態方程式と射影残差

落とした状態成分を $`r_t=(I_n-P^\top P)x_t`$ と置く。$`x_t=P^\top\xi_t+r_t`$ を元の状態方程式へ代入すると

```math
\xi_{t+1}=A_r\xi_t+B_ru_t+Pw_t+\varepsilon_t
```

となる。ここで

```math
A_r=PAP^\top,\qquad B_r=PB,\qquad
\varepsilon_t=PAr_t
```

である。射影残差が縮約モデルに入る効果は、

```math
\|\varepsilon_t\|\leq\|PA(I_n-P^\top P)\|\,\|r_t\|
```

で評価できる。これを元の高次元系の性能解析へつなげることが、現在進めている理論的な課題である。

### 5.3 縮約モデルの推定

入力と縮約状態の回帰ベクトルを

```math
\psi_t=\left[u_t^\top\;\xi_t^\top\right]^\top\in\mathbb R^{m+n_r}
```

とし、推定縮約モデルを $`\widehat\Theta_t^r=[\widehat B_t^r\quad\widehat A_t^r]`$ と書く。縮約データで初期モデルを推定した後、RLSを繰り返す。

```math
\widehat\Theta_{t+1}^r=\widehat\Theta_t^r+
\frac{\left(\xi_{t+1}-\widehat\Theta_t^r\psi_t\right)\psi_t^\top\Psi_t^{-1}}
{t+\psi_t^\top\Psi_t^{-1}\psi_t}
```

$`\Psi_t`$ は回帰データの標本共分散である。縮約空間に合わせてRLSの回帰次元を減らすことが、この構成の中心である。

### 5.4 低次元ゲインの更新

低次元ゲインを $`\bar K_t\in\mathbb R^{m\times n_r}`$ として

```math
u_t=\bar K_t\xi_t+e_t,\qquad K_t=\bar K_tP
```

とする。縮約評価重みを $`Q_r=PQP^\top`$、外乱共分散を $`W_r=PWP^\top`$ とし、推定閉ループ行列を $`\widehat A_{\bar K,t+1}^r=\widehat A_{t+1}^r+\widehat B_{t+1}^r\bar K_t`$ と置く。推定閉ループがSchur安定な更新時刻では、次の2本のLyapunov方程式を解く。

```math
\widehat\Sigma_{t+1}^r=W_r+
\widehat A_{\bar K,t+1}^r\widehat\Sigma_{t+1}^r
(\widehat A_{\bar K,t+1}^r)^\top
```

```math
\widehat S_{t+1}^r=Q_r+\bar K_t^\top R\bar K_t+
(\widehat A_{\bar K,t+1}^r)^\top\widehat S_{t+1}^r
\widehat A_{\bar K,t+1}^r
```

これらを用いた方策勾配と更新則は

```math
\widehat\nabla C_{t+1}^r(\bar K_t)=
2\left((R+(\widehat B_{t+1}^r)^\top\widehat S_{t+1}^r\widehat B_{t+1}^r)\bar K_t+
(\widehat B_{t+1}^r)^\top\widehat S_{t+1}^r\widehat A_{t+1}^r\right)
\widehat\Sigma_{t+1}^r
```

```math
\bar K_{t+1}=\bar K_t-\eta\,\widehat\nabla C_{t+1}^r(\bar K_t)
```

である。実装したゲイン $`K_t=\bar K_tP`$ は、評価時に100次元系へそのまま適用する。

### 5.5 計算規模の変化

| 計算対象 | Full | Reduced（24次元） |
|---|---:|---:|
| 回帰ベクトル | $`m+n=104`$ | $`m+n_r=28`$ |
| 推定モデルのパラメータ数 | $`n(n+m)=10400`$ | $`n_r(n_r+m)=672`$ |
| Lyapunov方程式の次数 | $`n=100`$ | $`n_r=24`$ |

推定と方策勾配の行列演算を小さい系へ移すことで、制御器のオンライン計算時間を短縮した。実際の処理時間は次節の数値実験で計測した。

## 6. PE条件と実験の評価指標

### 6.1 回帰データの条件

Fullの回帰行列 $`D_{0,t}`$ と縮約した回帰行列 $`D_{0,t}^r`$ は

```math
D_{0,t}=\left[\phi_0\;\cdots\;\phi_{t-1}\right],\qquad
\Phi_t=\frac1tD_{0,t}D_{0,t}^\top
```

```math
D_{0,t}^r=\left[\psi_0\;\cdots\;\psi_{t-1}\right],\qquad
\Psi_t=\frac1tD_{0,t}^r(D_{0,t}^r)^\top
```

と定義する。PE条件は、これらの標本共分散行列の最小特異値を用いて

```math
\sigma_{\min}(\Phi_t)\geq\gamma_t>0,\qquad
\sigma_{\min}(\Psi_t)\geq\gamma_t^r>0
```

と表せる。Fullでは104次元、24次元縮約では28次元の回帰を扱う。最小特異値 $`\sigma_{\min}`$ と条件数 $`\kappa=\sigma_{\max}/\sigma_{\min}`$ を比較し、弱励起方向が推定条件へ与える影響を調べた。

### 6.2 制御性能

真の100次元系のLQRコストで両手法を比較する。最適性ギャップを

```math
g_t=\frac{C(K_t)-C(K^\star)}{C(K^\star)}
```

とした。Reducedのゲインも $`K_t=\bar K_tP\in\mathbb R^{4\times100}`$ として100次元の状態に適用する。$`K^\star`$ は真のモデルからオフラインで求める比較用ゲインである。

### 6.3 制御器側オンライン計算時間

モデル推定、Lyapunov方程式、勾配計算、ゲイン更新に費やした時間を集計する。状態を進めるシミュレーションと真のモデルによる性能評価は、別処理として計時対象から外した。処理時間の単位はms/stepである。

### 6.4 対応するseed間の比較

同一seedで得たFullと24次元の最終最適性ギャップを1組として比較した。seed $`i`$ のギャップを $`g_i^{\mathrm{Full}}`$ と $`g_i^{24}`$ と書き、差を次式で定義する。

```math
d_i=g_i^{\mathrm{Full}}-g_i^{24},\qquad i=1,\ldots,10.
```

$`d_i>0`$ は24次元の方が小さいギャップを達成したことを表す。seed間で独立な試行を仮定し、**正確符号検定（両側）**で $`\Pr(d_i>0)=1/2`$ の帰無仮説を評価した。差がゼロの組はなく、10組すべてで $`d_i>0`$ だったため、正確なp値は

```math
p=2\left(\frac{1}{2}\right)^{10}=0.001953125
```

となる。この検定は差の正負を用いる。最適性ギャップの大きさとseed別の値は[比較表](../results/tables/comparison.md)にまとめた。

## 7. 比較実験と結果

| 項目 | 設定 |
|---|---|
| プラント | 100状態・4入力の合成離散時間線形系 |
| 手法 | Full-order indirect PGAC、Reduced-order indirect PGAC |
| 縮約次元 | $`n_r\in\{12,20,24,48,80\}`$ |
| 初期データ長 | $`t_0=600`$ |
| 実験時間長 | $`T=5600`$（オンライン区間5000ステップ） |
| 試行数 | 10 |

| 比較指標 | Full（100次元） | Reduced（24次元） |
|---|---:|---:|
| 最終最適性ギャップ（10試行の中央値） | 0.1399 | **0.0648** |
| 制御器側オンライン計算時間 | 9.43 ms/step | **0.184 ms/step** |
| 回帰共分散行列の最小特異値 | $`1.28\times10^{-12}`$ | **$`1.59\times10^{-4}`$** |
| 回帰共分散行列の条件数 | $`3.98\times10^9`$ | **$`32.4`$** |

この合成系では、24次元で重要な状態方向を確保しながら、モデル推定に含める弱励起方向を減らせた。10試行のギャップ平均はFullで **0.17428**、24次元で **0.07310** であり、24次元はすべてのseedでFullを上回った。24次元はオンライン計算時間と回帰条件でも良好な結果を示した。グラフは[結果一覧](../results/README.md)に掲載した。

## 8. 今後の研究

次の段階では、縮約次元の選定をデータから決める指針、射影残差を含むモデル推定誤差の解析、元の高次元閉ループ系での逐次安定性と制御性能の評価を進める。さらに、24方向の支配構造を持たないプラントや実データでも比較し、手法の適用範囲を調べる。

## 関連資料

- [研究の全体像と先行研究](research-overview.md)
- [数値実験の図と表](../results/README.md)
