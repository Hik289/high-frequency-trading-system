var username = $.cookie("username");
$("#username").text(username);
var layuimini;
layui.use(['element', 'layer', 'layuimini'], function () {
    var $ = layui.jquery,
        element = layui.element,
        layer = layui.layer;
        layuimini = layui.layuimini;
    $('.login-out').on("click", function () {
		$.cookie("token","", {path: '/'});
   	 	$.cookie("username","", {path: '/'});
   	 	$.cookie("type","", {path: '/'});
        layer.msg('login out success', function () {
            window.location = '/login';
        });
    });
    layuimini.listen();
});
var refreshTab = function(){
	layuimini.refresh();
// 	$(".layui-tab-item.layui-show").find("iframe")[0].contentWindow.location.reload();
}
var goUrl = function(url){
	location.href=url
}
var addTeamUser = function(teamId){
	if(teamId==null||teamId==""){
		layer.msg("Please add team first");
		return false;
	}
	$.post('/private/team/addTeamUser', {teamId:teamId}, function(str){
	  layer.open({
	    type: 1,
	    title:"Invite Team Members",
	    offset: '80px',
	    area: ['400px', '300px'],
	    content: str 
	  });
	}); 
}
var removeUser = function(userId,teamId){
	$.post("/private/team/removeUser",{userId:userId,teamId:teamId},function(obj){
		layer.msg(obj.message,function(){
			if(obj.success){
				window.location.reload(); 
			}
		});
	},"json");
}
var addBoard = function(teamId){
	if(teamId==null||teamId==""){
		layer.msg("Please add team first");
		return false;
	}
	$.post('/private/board/addBoard', {teamId:teamId}, function(str){
	  layer.open({
	    type: 1,
	    title:"Add Board",
	    offset: '80px',
	    area: ['400px', '300px'],
	    content: str 
	  });
	}); 
}
var board = function(boardId){
//	goUrl("#/page/team-board.html?boardId="+boardId);
	goUrl("/team/board?boardId="+boardId);
}
function getUrlParam(name) {
	var l = window.location.hash.split("?")[1];
    var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
    var r = l.match(reg);
    if (r != null) {
        return decodeURI(r[2])
    } else {
        return null
    }
}