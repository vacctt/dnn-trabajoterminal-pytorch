<h1>VADCC</h1>
<p>Este repositorio contiene el código fuente del modelo utilizado por la Raspberry Pi 4B para predecir:</p>
<ul>
    <li>Somnolencia</li>
    <li>Sin somnolencia</li>
</ul>
El modelo fue entrenado con el archivo <b>model.py</b> usando la clase <em>ModelMobileNetv2</em>

Utiliza la arquitectura MobileNetV2 con el framework PyTorch.

El archivo <b>plot.py</b> nos permite hacer gráficas haciendo uso de la librería MatplotLib importando la clase <em>Plotter</em>

El archivo <b>camera.py </b> nos permite hacer uso de nuestra cámara de la computadora y predecir lo que ésta capta importando la función <em>start</em>. (Nota: es necesario contar con el modelo entrenado que sale como resultado de usar el archivo <b>model.py</b> ) 

