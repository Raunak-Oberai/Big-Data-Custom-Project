
function validate_empty(elementid)
{
    var textValue = document.getElementById(elementid).value;

    if (!textValue)
    {
        alert("Please Enter Search Tag!");
        return false;
    }
    return true;
}