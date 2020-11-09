function ajaxCall(url, params, destination)
{
			var xhttp = new XMLHttpRequest();
			xhttp.onreadystatechange = function () {
				if (this.readyState == 4 && this.status == 200)
				{

			    document.getElementById(destination).innerHTML = this.responseText;
				}
			};
			xhttp.open("POST", url, true);
			xhttp.send(params);
}
