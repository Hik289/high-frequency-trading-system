var token = $.cookie("token");
var checkToken = function(){
	if(token!=null&&token!=""){
		$.ajax({
			url:'/login/checkToken',
			type:'POST',
			dataType:'json',
			async:false,
			data:{token:token},
			success:function(obj){
				if(!obj.success){
					$.cookie("token","", {path: '/'});
			   	 	$.cookie("username","", {path: '/'});
			   	 	$.cookie("type","", {path: '/'});
			   	 	window.location.href = "login";
				}
			}
		});
	}else{
		$.cookie("token","", {path: '/'});
   	 	$.cookie("username","", {path: '/'});
   	 	$.cookie("type","", {path: '/'});
   	 	window.location.href = "Login.html";
	}
}
checkToken();