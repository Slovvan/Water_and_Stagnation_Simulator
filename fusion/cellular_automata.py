from water.soil_terrain import generate_soil_terrain
from water.water_class import water_cellular_automata
from water.water_animation import animate


def main():
    grid_size = 200
    dz_mm = 10.0   # 1 celda = 10 mm  ->  corte de 2 m x 2 m
    dt_s = 10.0    # con dt = 60 s la infiltración sale 5-14 % baja frente a Green-Ampt (ver documento, 6.3)

    soil = generate_soil_terrain(grid_size=grid_size, dz_mm=dz_mm, seed=3, ks_sigma_ln=0.5, bedrock="fractured")
    print(soil.summary())

    horas_tormenta, horas_total = 3.0, 10.0
    simulation = water_cellular_automata(
        grid_size=grid_size,
        time_steps=int(horas_total * 3600 / dt_s),   # 10 h simuladas
        rain_steps=int(horas_tormenta * 3600 / dt_s),  # llueve 3 h y luego se ve cómo drena / se estanca
        rain_intensity_mm_h=30.0,
        rain_noise=0.3,
        rain_seed=40,
        dz_mm=dz_mm,
        dt_s=dt_s,
        substeps=4,
        soil=soil,
        soil_initial_saturation=0.30,
        edge_runoff=True,
        bottom_drain=True,
    )
    animate(simulation, steps_per_frame=8)


if __name__ == "__main__":
    main()
