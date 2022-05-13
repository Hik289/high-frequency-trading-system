package com.boot.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;

import com.boot.entity.Board;
import com.boot.entity.BoardList;
import com.boot.entity.Team;
import com.boot.entity.User;
import com.boot.service.BoardService;
import com.boot.service.TeamService;
import com.boot.vo.TokenHandler;
@Controller
public class ManageController {
	@Autowired
	private TeamService teamService;
	@Autowired
	private BoardService boardService;
	/**
	 * index
	 */
	@RequestMapping("/index")
	public String index(ModelMap model,Integer teamId) {
		//team 信息
		Team team = null;
		if(teamId==null) {
			//获取用户默认的一个team
			Integer userId = TokenHandler.getBusinessId();
			team = teamService.getTeamDefault(userId);
		}
		//获取team 人员
		List<User> userList = null;
		//获取board
		List<Board> boardList = null;
		if(teamId!=null) {
			team = teamService.getDeatil(teamId);
		}
		if(team!=null) {
			userList = teamService.getTeamUser(team.getId());
			boardList = boardService.getListByTeam(team.getId());
		}
		model.put("team", team);
		model.put("userList", userList);
		model.put("boardList", boardList);
		return "index";
	}
	
	/**
	 * team board
	 * @param model
	 * @return
	 */
	@RequestMapping("/team/board")
	public String teamboard(ModelMap model,Integer boardId) {
		//get board list
		List<BoardList> boardList = boardService.getBoardListByBoard(boardId);
		model.put("boardList", boardList);
		if(boardList!=null&&boardList.size()>0) {
			for(BoardList bl : boardList) {
				bl.setListCard(boardService.getBoardListCard(bl.getId()));
			}
		}
		Board board = boardService.getDetailByid(boardId);
		model.put("board", board);
		model.put("boardId", boardId);
		//team
		
		return "page/team-board";
	}
}