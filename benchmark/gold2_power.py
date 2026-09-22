#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analisis de potencia: cuantas placas necesita el nuevo gold limpio.

El gold actual solo aporta 126 placas no vistas en entrenamiento, y con ese tamano la
comparacion estudiante-vs-maestro no tiene potencia para resolver diferencias de 1-2 pp
(p = 1.00 con solo 5 pares discordantes). La pregunta practica es cuantas placas hay que
anotar para que la prueba concluya algo.

La potencia de McNemar NO depende del tamano de muestra sin mas, sino del numero esperado
de pares DISCORDANTES: los pares en que ambos modelos aciertan o ambos fallan no aportan
informacion. Con exactitudes del orden del 95 % la discordancia es rara (~4 %), asi que
hacen falta bastantes placas para acumular discordantes suficientes.

Se simula bajo tres escenarios de diferencia real, porque la estimacion puntual del gold
actual (basada en 5 discordantes) es demasiado ruidosa para planificar con ella.

Uso:
    python gold2_power.py
"""
import argparse, math

import numpy as np


def p_mcnemar_exacta(b10, b01):
    n = b10 + b01
    if n == 0:
        return 1.0
    k = min(b10, b01)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cola)


def potencia(n, p_disc, q, n_sim=4000, alfa=0.05, seed=42):
    """Fraccion de simulaciones en que McNemar detecta la diferencia.

    n       placas anotadas
    p_disc  probabilidad de que una placa sea discordante entre los dos modelos
    q       probabilidad de que, siendo discordante, gane el modelo A
    """
    rng = np.random.default_rng(seed)
    n_disc = rng.binomial(n, p_disc, size=n_sim)
    b10 = rng.binomial(n_disc, q)
    b01 = n_disc - b10
    exitos = sum(p_mcnemar_exacta(int(a), int(b)) < alfa for a, b in zip(b10, b01))
    return exitos / n_sim


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-disc", type=float, default=0.04,
                    help="tasa de discordancia esperada entre los dos sistemas")
    ap.add_argument("--sim", type=int, default=4000)
    args = ap.parse_args()

    # Escenarios: diferencia real neta = p_disc * (2q - 1)
    escenarios = []
    for dif in (0.01, 0.02, 0.03):
        q = (dif / args.p_disc + 1) / 2
        if 0.5 < q <= 1.0:
            escenarios.append((dif, q))

    tamanos = [126, 300, 500, 800, 1000, 1500, 2000, 2500, 3000]

    print(f"Tasa de discordancia asumida: {args.p_disc:.1%} "
          f"(observada en el gold actual: 26/782 = 3.3 %)\n")
    cab = "  ".join(f"dif={d:.0%}".rjust(7) for d, _ in escenarios)
    print(f"{'n placas':>9s}  {'discord.':>9s}  {cab}")
    for n in tamanos:
        esperados = n * args.p_disc
        fila = "  ".join(f"{potencia(n, args.p_disc, q, args.sim):7.2f}" for _, q in escenarios)
        marca = "   <- gold limpio actual" if n == 126 else ""
        print(f"{n:9d}  {esperados:9.1f}  {fila}{marca}")

    print("\nLectura: cada celda es la probabilidad de detectar la diferencia al 5 %.")
    print("Se busca >= 0.80. dif es la diferencia real de exactitud entre los dos sistemas.")

    # Tamano minimo recomendado para el caso de interes (2 pp)
    objetivo = next(((d, q) for d, q in escenarios if abs(d - 0.02) < 1e-9), None)
    if objetivo:
        d, q = objetivo
        n = 200
        while n <= 6000 and potencia(n, args.p_disc, q, args.sim) < 0.80:
            n += 100
        print(f"\nPara dif = {d:.0%} con potencia 0.80 hacen falta ~{n} placas anotadas.")


if __name__ == "__main__":
    main()
