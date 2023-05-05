const apiUrl = "http://127.0.0.1:5000"

export const getRegex = async (matches, rejections) => {
    if (!matches.length > 0) {
        alert("Please Add Match Texts");
        return;
    }
    if (!rejections.length > 0) {
        alert("Please Add Rejection Texts");
        return;
    }
    let request = await fetch(`${apiUrl}/get-regex`, {
        method: "POST",
        headers: {
            "Content-Type": "text/html",
        },
        body: JSON.stringify({
            matches: matches,
            rejections: rejections,
        })
    })

    const response = await request.json();

    return response;
}