import matplotlib.pyplot as plt
import seaborn as sns

arquivo = "synthetic_health_data.csv"
sep = ","
alvo = "Health_Score"
peso_inicial = 1.0
alpha = 0.1
iteracoes = 200
k = 5
semente = 17
normalizar = True


def ler_dados(caminho):
  with open(caminho, "r", encoding="utf-8") as f:
    linhas = f.read().strip().split("\n")

  colunas = [c.strip().strip('"') for c in linhas[0].split(sep)]
  for c in range(len(colunas)):
    if colunas[c] == alvo:
      pos_y = c

  X, y = [], []
  for linha in linhas[1:]:
    valores = [v.strip().strip('"') for v in linha.split(sep)]
    x = []
    for c in range(len(valores)):
      if c == pos_y:
        y.append(float(valores[c]))
      else:
        x.append(float(valores[c]))
    X.append(x)

  features = [c for c in colunas if c != alvo]
  return X, y, features


def media(v):
  total = 0.0
  for num in v:
    total += num
  return total / len(v)


def desvio(v):
  m = media(v)
  total = 0.0
  for num in v:
    total += (num - m) ** 2
  return (total / (len(v) - 1)) ** 0.5


def normalizacao(X):
  medias, desvios = [], []
  for j in range(len(X[0])):
    coluna = [linha[j] for linha in X]
    medias.append(media(coluna))
    desvios.append(desvio(coluna))
  return medias, desvios


def normaliza(X, medias, desvios):
  return [[(linha[j] - medias[j]) / desvios[j] for j in range(len(linha))] for linha in X]


def com_x0(X):
  return [[1.0] + linha for linha in X]


def predizer(X, beta):
  preds = []
  for linha in X:
    soma = 0.0
    for j in range(len(beta)):
      soma += beta[j] * linha[j]
    preds.append(soma)
  return preds


def custo(preds, y):
  m = len(y)
  total = 0.0
  for i in range(m):
    total += (preds[i] - y[i]) ** 2
  return total / (2 * m)


def gradientes(X, preds, y):
  m = len(y)
  grads = []
  for j in range(len(X[0])):
    total = 0.0
    for i in range(m):
      total += (preds[i] - y[i]) * X[i][j]
    grads.append(total / m)
  return grads


def treinar(X, y):
  beta = [peso_inicial] * len(X[0])
  historico = []

  for it in range(iteracoes):
    preds = predizer(X, beta)
    J = custo(preds, y)
    grads = gradientes(X, preds, y)
    beta = [beta[j] - alpha * grads[j] for j in range(len(beta))]
    historico.append(J)

  return beta, historico


def rmse(y, preds):
  total = 0.0
  for j in range(len(y)):
    total += (y[j] - preds[j]) ** 2
  return (total / len(y)) ** 0.5


def r2(y, preds):
  m = media(y)
  res, tot = 0.0, 0.0
  for j in range(len(y)):
    res += (y[j] - preds[j]) ** 2
    tot += (y[j] - m) ** 2
  return 1 - res / tot


def gerar_folds(n):
  idx = list(range(n))
  estado = semente
  for i in range(n - 1, 0, -1):
    estado = estado * 2
    j = estado % (i + 1)
    idx[i], idx[j] = idx[j], idx[i]

  folds = []
  inicio = 0
  for f in range(k):
    tam = n // k + (1 if f < n % k else 0)
    folds.append(idx[inicio:inicio + tam])
    inicio += tam
  return folds


X, y, features = ler_dados(arquivo)
print(f"Base: {len(X)} exemplos, {len(features)} características")

folds = gerar_folds(len(y))
lista_rmse, lista_r2 = [], []
reais, preditos, cores = [], [], []
curvas = []

for f in range(k):
  teste = folds[f]
  treino = [i for g in range(k) if g != f for i in folds[g]]

  X_treino = [X[i] for i in treino]
  y_treino = [y[i] for i in treino]
  X_teste = [X[i] for i in teste]
  y_teste = [y[i] for i in teste]

  if normalizar:
    medias, desvios = normalizacao(X_treino)
    X_treino = normaliza(X_treino, medias, desvios)
    X_teste = normaliza(X_teste, medias, desvios)

  X_treino = com_x0(X_treino)
  X_teste = com_x0(X_teste)

  print(f"Fold {f + 1}")
  beta, historico = treinar(X_treino, y_treino)
  curvas.append(historico)

  preds = predizer(X_teste, beta)
  erro = rmse(y_teste, preds)
  acerto = r2(y_teste, preds)

  lista_rmse.append(erro)
  lista_r2.append(acerto)

  reais += y_teste
  preditos += preds
  cores += [f"Fold {f + 1}"] * len(y_teste)

  print(f"   Custo final = {historico[-1]:.4f} | RMSE = {erro:.4f} | R² = {acerto:.4f}\n")

print("===== Resultado do K-fold =====")
print(f"RMSE: média = {media(lista_rmse):.4f} | desvio padrão = {desvio(lista_rmse):.4f}")
print(f"R²  : média = {media(lista_r2):.4f} | desvio padrão = {desvio(lista_r2):.4f}")

menor, maior = reais[0], reais[0]
for v in reais + preditos:
  menor = v if v < menor else menor
  maior = v if v > maior else maior

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

sns.scatterplot(x=reais, y=preditos, hue=cores, ax=ax[0])
ax[0].plot([menor, maior], [menor, maior], "k--", label="Previsão perfeita")
ax[0].set_title("Valores reais x preditos")
ax[0].set_xlabel("Y real")
ax[0].set_ylabel("Y predito")
ax[0].legend()

for f in range(len(curvas)):
  ax[1].plot(curvas[f], label=f"Fold {f + 1}")
ax[1].set_title("Custo J(β) por iteração")
ax[1].set_xlabel("Iteração")
ax[1].set_ylabel("J(β)")
ax[1].set_yscale("log")
ax[1].legend()

plt.tight_layout()
plt.savefig("resultados.png", dpi=150)
plt.show()